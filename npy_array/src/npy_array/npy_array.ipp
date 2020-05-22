#include "npy_array/npy_array.h"

template<typename Iterator>
size_t multiplies_vector(const Iterator start, const Iterator end)
{
    return std::accumulate(start, end, size_t(1), std::multiplies<size_t>());
}

template<class T> 
bool npy_array<T>::check_magic_string(std::ifstream &array_file)
{
    std::string magic_string{};
    magic_string.resize(6);

    array_file.read(&magic_string[0], magic_string.size());

    return magic_string == "\x93NUMPY";
}

template<class T> 
bool npy_array<T>::check_version(std::ifstream &array_file)
{
    uint8_t major_version;
    uint8_t minor_version;

    array_file.read(reinterpret_cast<char*>(&major_version), 1);
    array_file.read(reinterpret_cast<char*>(&minor_version), 1);

    return major_version == 0x1;
}

template<class T> 
std::string npy_array<T>::read_header(std::ifstream &array_file)
{
    std::string header;
    uint16_t header_length;

    array_file.read(reinterpret_cast<char*>(&header_length), 2);

    header.resize(header_length);

    array_file.read(&header[0], header_length);

    return header;
}

template<typename T>
void npy_array<T>::parse_header(const std::string& header)
{
    std::array<char, 5> charaters_to_remove{' ', '\n', '{', '}', '\''};
    // Create a copy of the header string because we need to modify it.
    auto header_copy{header};
    // Remove the unwanted characters from the header copy string.
    header_copy.erase(std::remove_if(header_copy.begin(), header_copy.end(), [&](char c)
    {
        return std::find(charaters_to_remove.cbegin(), charaters_to_remove.cend(), c) != charaters_to_remove.cend();
    }), header_copy.end());

    // Find the indexes of the left and right brackes within the header string.
    // Both key value pairs and shape elements are comma-separeted.
    // We need to change the separation of the shape by replacing the comma in the shape string with another token that easily recognizable, like an underscore.
    auto left_bracket_index = std::find(header_copy.begin(), header_copy.end(), '(');
    auto right_bracket_index = std::find(header_copy.begin(), header_copy.end(), ')');

    if(left_bracket_index != header_copy.end() && right_bracket_index != header_copy.end())
    {
        std::replace(left_bracket_index, right_bracket_index, ',', '_');
    }
    else
    {
        throw boost::regex_error{boost::regex_constants::error_unknown};
    }
    

    std::vector<std::string> header_key_value_pairs;

    boost::split(header_key_value_pairs, header_copy, [](char c){return c == ',';});

    for(auto& key_value_pair : header_key_value_pairs)
    {
        std::vector<std::string> key_value{};
        boost::split(key_value, key_value_pair, [](char c){return c == ':';});

        if(key_value[0] == "descr")
        {
            npy_dtype requested_dtype = npy_dtype::from_string(key_value[1]);

            if(requested_dtype && npy_dtype::from_type<T>() == requested_dtype)
            {
                _dtype = std::move(requested_dtype);
            }
            else
            {
                throw boost::regex_error{boost::regex_constants::error_unknown};
            }
            
        }
        else if(key_value[0] == "fortran_order")
        {
            if(!(key_value[1] == "False" || key_value[1] == "True"))
            {
                throw boost::regex_error{boost::regex_constants::error_unknown};
            }

            _fortran_order = key_value[1] == "False" ? false : true;
        }
        else if(key_value[0] == "shape")
        {
            std::vector<std::string> shape_values{};
            boost::split(shape_values, key_value[1].substr(1, key_value[1].size() - 2), [](char c){return c == '_';});
            shape_values.erase(std::remove_if(shape_values.begin(), shape_values.end(), [](std::string& s)
            {
                return s.empty();
            }), shape_values.end());
            _shape.reserve(shape_values.size());
            std::transform(shape_values.cbegin(), shape_values.cend(), std::back_inserter(_shape), [](std::string shape_value) -> size_t
            {
                return std::strtoul(shape_value.data(), nullptr, 10);
            });

            if(std::any_of(_shape.cbegin(), _shape.cend(), [](size_t d){return d == 0;}))
            {
                throw boost::regex_error{boost::regex_constants::error_unknown};
            }
        }
        else if(key_value[0] == "") continue;
        else
        {
            throw boost::regex_error{boost::regex_constants::error_unknown};
        }
        
    }
}


template<typename T>
npy_array<T>::npy_array(const std::string& array_path)
    : _shape{}, _data{}, _dtype{}, _fortran_order{false}
{
    std::ifstream array_file{};
    std::string header{};

    array_file.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);

    try
    {
        array_file.open(array_path);

        if(!this->check_magic_string(array_file))
        {
            throw npy_array_exception{npy_array_exception_type::invalid_magic_string};
        }

        if(!this->check_version(array_file))
        {
            throw npy_array_exception{npy_array_exception_type::unsupported_version};
        }

        this->parse_header(this->read_header(array_file));

        _data.resize(multiplies_vector(_shape.cbegin(), _shape.cend()));
        array_file.read(reinterpret_cast<char*>(_data.data()), _data.size() * sizeof(T));
    }
    catch(const std::ios_base::failure& failure_exception)
    {
        throw npy_array_exception{npy_array_exception_type::input_output_error};
    }
    catch(const boost::regex_error& regex_exception)
    {
        throw npy_array_exception{npy_array_exception_type::ill_formed_header};
    }
    catch(const std::bad_alloc& bad_alloc_exception)
    {
        throw npy_array_exception{npy_array_exception_type::unsufficient_memory};
    }
    catch(const std::exception& exeption)
    {
        throw npy_array_exception{npy_array_exception_type::generic};
    }
}

template<typename T>
npy_array<T>::npy_array(const std::vector<size_t>& shape) 
    : _shape{shape}, _data{}, _dtype{}, _fortran_order{false}
{
    npy_dtype requested_dtype = npy_dtype::from_type<T>();

    if(requested_dtype)
    {
        _dtype = std::move(requested_dtype);
    }
    else
    {
        throw npy_array_exception{npy_array_exception_type::unsupported_dtype};
    }
    
    _data.resize(multiplies_vector(_shape.cbegin(), _shape.cend()));
}

template<typename T>
npy_array<T>::npy_array(std::vector<size_t>&& shape)
    : _shape{std::move(shape)}, _data{}, _dtype{}, _fortran_order{false}
{
    npy_dtype requested_dtype = npy_dtype::from_type<T>();

    if(requested_dtype)
    {
        _dtype = std::move(requested_dtype);
    }
    else
    {
        throw npy_array_exception{npy_array_exception_type::unsupported_dtype};
    }

    _data.resize(multiplies_vector(_shape.cbegin(), _shape.cend()));
}

template<class T> 
npy_array<T>::npy_array(std::initializer_list<size_t> shape_list)
    : _shape{shape_list}, _data{}, _dtype{}, _fortran_order{false}
{
    npy_dtype requested_dtype = npy_dtype::from_type<T>();

    if(requested_dtype)
    {
        _dtype = std::move(requested_dtype);
    }
    else
    {
        throw npy_array_exception{npy_array_exception_type::unsupported_dtype};
    }
    
    _data.resize(multiplies_vector(_shape.cbegin(), _shape.cend()));
}


template<typename T>
npy_array<T>::npy_array(const std::vector<size_t>& shape, const std::vector<T>& data)
    : _shape{}, _data{}, _dtype{}, _fortran_order{false}
{
    if(multiplies_vector(shape.cbegin(), shape.cend()) == data.size())
    {
        npy_dtype requested_dtype = npy_dtype::from_type<T>();

        if(requested_dtype)
        {
            _dtype = std::move(requested_dtype);
        }
        else
        {
            throw npy_array_exception{npy_array_exception_type::unsupported_dtype};
        }
        
        _shape = shape;
        _data = data;
    }
    else
    {
        throw npy_array_exception{npy_array_exception_type::unmatched_shape_data};
    }
    
} 

template<typename T>
npy_array<T>::npy_array(std::vector<size_t>&& shape, std::vector<T>&& data)
    : _shape{}, _data{}, _dtype{}, _fortran_order{false}
{
    if(multiplies_vector(shape.cbegin(), shape.cend()) == data.size())
    {
        npy_dtype requested_dtype = npy_dtype::from_type<T>();

        if(requested_dtype)
        {
            _dtype = std::move(requested_dtype);
        }
        else
        {
            throw npy_array_exception{npy_array_exception_type::unsupported_dtype};
        }
        

        _shape = std::move(shape);
        _data = std::move(data);
    }
    else
    {
         throw npy_array_exception{npy_array_exception_type::unmatched_shape_data};
    }
    
}

template<class T> 
npy_array<T>::npy_array(std::initializer_list<size_t> shape_list, std::initializer_list<T> data_list)
    : _shape{}, _data{}, _dtype{}, _fortran_order{false}
{
    if(multiplies_vector(shape_list.begin(), shape_list.end()) == data_list.size())
    {
        npy_dtype requested_dtype = npy_dtype::from_type<T>();

        if(requested_dtype)
        {
            _dtype = std::move(requested_dtype);
        }
        else
        {
            throw npy_array_exception{npy_array_exception_type::unsupported_dtype};
        }
        

        _shape = shape_list;
        _data = data_list;
    }
    else
    {
         throw npy_array_exception{npy_array_exception_type::unmatched_shape_data};
    }
}

template<class T> 
T& npy_array<T>::operator[](size_t index)
{
    if(index >= 0 && index < this->size()) return _data[index];
    else
    {
        std::ostringstream string_stream{};
        string_stream << "The index " << index << " is out of range (0, " << this->size() << ")";
        throw std::out_of_range{string_stream.str()};
    } 
}

template<class T> 
const T& npy_array<T>::operator[](size_t index) const
{
    if(index >= 0 && index < this->size()) return _data[index];
    else
    {
        std::ostringstream string_stream{};
        string_stream << "The index " << index << " is out of range (0, " << this->size() << ")";
        throw std::out_of_range{string_stream.str()};
    }
}



template<class T> const std::vector<size_t> &npy_array<T>::shape() const noexcept {return _shape;}
template<class T> const npy_dtype &npy_array<T>::dtype() const noexcept {return _dtype;}
template<class T> bool npy_array<T>::fortran_order() const noexcept {return _fortran_order;}

template<class T> const T *npy_array<T>::data() const noexcept {return _data.data();}
template<class T> T *npy_array<T>::data() noexcept {return _data.data();}

template<typename T> size_t npy_array<T>::size() const noexcept {return _data.size();}
template<typename T> size_t npy_array<T>::byte_size() const noexcept {return _data.size() * sizeof(T);}

template<class T> 
void npy_array<T>::save(const std::string &array_path)
{
    std::stringstream header_string{};
    header_string << "{'descr': '" << _dtype.str() << "', 'fortran_order': ";
    header_string << (_fortran_order ? "True" : "False");
    header_string << ", 'shape': (";

    for(size_t i = 0; i != _shape.size() - 1; i++)
    {
        header_string << _shape[i] << ", ";
    }

    header_string << *(_shape.cend() - 1) << "), }";

    size_t header_size = 6 + 2 + 2 + header_string.str().size() + 1;
    size_t padding = 64 - (header_size % 64);

    if(padding > 0)
    {
        header_string << std::string(padding, '\x20');
        header_string << '\n';
        header_size += padding;
    }

    header_size = header_string.str().size();

    std::ofstream array_stream{array_path, std::ios_base::out | std::ios_base::binary};
    array_stream << "\x93NUMPY";
    array_stream << uint8_t(0x01) << uint8_t(0x00);
    array_stream.write(reinterpret_cast<const char*>(&header_size), sizeof(uint16_t));
    //array_stream << uint16_t(header_string.str().size());
    array_stream << header_string.str();
    array_stream.write(reinterpret_cast<const char*>(_data.data()), this->byte_size());
    array_stream.flush();
}



