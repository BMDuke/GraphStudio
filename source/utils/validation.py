import struct
from crcmod.predefined import mkPredefinedCrcFun

'''
This is a collection of utilities which can be used to validate 
data assets
'''


# SOURCED FROM: 


_crc_fn = mkPredefinedCrcFun('crc-32c')


def calc_masked_crc(data):
    '''
    '''
    crc = _crc_fn(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xFFFFFFFF


def validate_dataset_slower(filenames, verbose=False):
    '''
    Code adapted from:
    https://gist.github.com/ed-alertedh/9f49bfc6216585f520c7c7723d20d951

    Credit: 
    Edward Bordin
    '''
    total_records = 0
    total_bad_len_crc = 0
    total_bad_data_crc = 0
    for f_name in filenames:
        i = 0

        if verbose:
            print('validating ', f_name)

        with open(f_name, 'rb') as f:

            len_bytes = f.read(8)
            while len(len_bytes) > 0:
                # tfrecord format is a wrapper around protobuf data
                length, = struct.unpack('<Q', len_bytes) # u64: length of the protobuf data (excluding the header)
                len_crc, = struct.unpack('<I', f.read(4)) # u32: masked crc32c of the length bytes
                data = f.read(length) # protobuf data
                data_crc, = struct.unpack('<I', f.read(4)) # u32: masked crc32c of the protobuf data

                if len_crc != calc_masked_crc(len_bytes):

                    if verbose:
                        print('bad crc on len at record', i)

                    total_bad_len_crc += 1

                if data_crc != calc_masked_crc(data):

                    if verbose:
                        print('bad crc on data at record', i)
                    
                    total_bad_data_crc += 1

                i += 1
                len_bytes = f.read(8)

        if verbose:
            print('checked', i, 'records')

        total_records += i
    
    if verbose:
        print('checked', total_records, 'total records')
        print('total with bad length crc: ', total_bad_len_crc)
        print('total with bad data crc: ', total_bad_data_crc)

    return total_records, total_bad_len_crc, total_bad_data_crc
    
