import struct

def write_int8(file, x : int) :
    a = struct.pack('I', x)
    file.write(a)
if __name__ == "__main__":
    file = open("Light.theme", "wb")
    
    _backgrouind_col = 0x776E65
    _text_col = 0x000000
    _tile_text_col = 0xF9F7Ef

    write_int8(file, _backgrouind_col)
    write_int8(file, _text_col)
    write_int8(file, _tile_text_col)

    tile_col = [0x8F857A, 0xB0AB9A, 0xD5C9B7, 0xF69664, 0xF77E61, 0xF75F3B, 0xEDD073, 0xF0D591]

    for i in range(0, 15):
        if i < len(tile_col): write_int8(file, tile_col[i])
        else: write_int8(file, tile_col[-1])