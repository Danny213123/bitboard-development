import numpy as np
import chess as ch

from chess_constants import File, HOT, Square, Rank, DARK_SQUARES, LIGHT_SQUARES, piece_to_glyph, BOARD_SQUARES

def set_bit(bitboard: np.uint64, bit: int) -> np.uint64:
    """
    Sets a bit in the provided unsigned 64-bit integer bitboard representation to 1
    :param bitboard: np.uint64 number
    :param bit: the binary index to turn hot
    :return: a copy of the bitboard with the specified `bit` set to 1
    """
    return np.uint64(bitboard | np.uint64(1) << np.uint64(bit))


def clear_bit(bitboard: np.uint64, bit: int or np.uint64) -> np.uint64:
    """
    Sets a bit in the provided unsigned 64-bit integer bitboard representation to 0
    :param bitboard: np.uint64 number
    :param bit: the binary index to turn off
    :return: a copy of the bitboard with the specified `bit` set to 0
    """
    return bitboard & ~(np.uint64(1) << np.uint64(bit))

##############################################################################################################
#                                                                                                            #  
#   The following functions are used to print the bitboards in a human readable format.                      #
#                                                                                                            #
##############################################################################################################

def print_bitboard(bitboard: np.uint64) -> None:
    """
    Prints the bitboard in a human readable format.
    :param bitboard: uint64 bitboard
    :return: None
    """

    print("   a b c d e f g h")
    print("  -----------------")
    for rank in range(8):
        print(8 - rank, end=" | ")
        for file in range(8):
            square = rank * 8 + file
            piece = bitboard[rank][file]
            if piece == 1:
                print("p", end=" ")
            else:
                print(".", end=" ")
        print("|")
    print("  -----------------")

def print_entire_board(bitboards: list):
    """
    Prints the entire board in a human readable format.
    :param bit_boards: list of uint64 bitboards
    :return: None
    """

    piece_symbols = {
        0: "p", 1: "n", 2: "b", 3: "r", 4: "q", 5: "k",
        6: "P", 7: "N", 8: "B", 9: "R", 10: "Q", 11: "K"
    }

    main_board = [["." for _ in range(8)] for _ in range(8)]

    for bit_board in enumerate(bitboards):
        for rank in range(8):
            for file in range(8):
                if bit_board[1][rank][file] == 1:
                    main_board[rank][file] = piece_symbols[bit_board[0]]

    print("   a b c d e f g h")
    print("  -----------------")
    for rank in range(8):
        print(8 - rank, end=" | ")
        for file in range(8):
            print(main_board[rank][file], end=" ")
        print("|")

##############################################################################################################
#                                                                                                            #
#   The following functions are used to convert bitboards to arrays and vice versa.                          #
#                                                                                                            #
##############################################################################################################

def bitboard_to_array(bit_board: np.uint64) -> np.ndarray:
    """
    Converts uint64 bitboard to a 8x8 numpy array.
    :param bit_board: uint64 bitboard
    :return: 8x8 numpy array
    """

    # create a set of 8 bit masks
    set = 8 * np.arange(7, -1, -1, dtype=np.uint64)

    # shift the bitboard by the set
    bit = (bit_board >> set).astype(np.uint8)

    # unpack the bits
    bit = np.unpackbits(bit, bitorder="little")

    return bit.reshape(8, 8)
    
def bitboards_to_array(bit_boards: list) -> np.ndarray:
    """
    Converts a list of uint64 bitboards to a 12x8x8 numpy array.
    :param bit_boards: list of uint64 bitboards
    :return: 12x8x8 numpy array
    """

    # convert the list to a numpy array
    bit_boards = np.asarray(bit_boards, dtype=np.uint64)[:, np.newaxis]

    # create a set of 8 bit masks
    set = 8 * np.arange(7, -1, -1, dtype=np.uint64)

    # shift the bitboard by the set
    bit = (bit_boards >> set).astype(np.uint8)

    # unpack the bits
    bit = np.unpackbits(bit, bitorder="little")

    return bit.reshape(-1, 8, 8)

##############################################################################################################
#                                                                                                            #
#   The Knight                                                                                               #
#                                                                                                            #
##############################################################################################################

def make_knight_attack(bitboard: np.uint64) -> np.uint64:
    
    knight_attack = np.uint64(0)

    for square in get_knight_squares(bitboard):
        knight_attack |= gen_knight_attack(square)
    return knight_attack

def get_knight_squares(bitboard: np.uint64) -> int:

    knight_squares = []

    for square in range(64):
        if bitboard & np.uint64(1) << np.uint64(square):
            knight_squares.append(square)

    return knight_squares

def gen_knight_attack(square: int) -> np.uint64:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static knight attack patterns
    :return: dict {square: knight attack bitboard} static square -> bitboard mapping
    """
    attack_bitboard = np.uint64(0)
    #print(square)

    for shift in [6, 10, 15, 17, -6, -10, -15, -17]:
        attack_bitboard |= set_bit(attack_bitboard, square+shift)

        if square in (File.A | File.B):
            attack_bitboard &= ~(np.uint64(File.hexG | File.hexH))
        if square in (File.G | File.H):
            attack_bitboard &= ~(np.uint64(File.hexA | File.hexB))

    return attack_bitboard


##############################################################################################################
#                                                                                                            #
#   The Bishop                                                                                               #
#                                                                                                            #
##############################################################################################################

def make_bishop_attack(bitboard: np.uint64) -> np.uint64:
        
    bishop_attack = np.uint64(0)

    for square in get_bishop_squares(bitboard):
        bishop_attack |= gen_bishop_attack(square)
    return bishop_attack

def get_bishop_squares(bitboard: np.uint64) -> int:

    bishop_squares = []

    for square in range(64):
        if bitboard & np.uint64(1) << np.uint64(square):
            bishop_squares.append(square)

    return bishop_squares

def gen_bishop_attack(from_square: int) -> np.uint64:
    """
    Generates all diagonal attacks from the provided square on an otherwise empty bitboard
    :param from_square: starting square from which to generate diagonal attacks
    :return: np.uint64 diagonal attacks bitboard from the provided square
    """
    attack_bb = np.uint64(0)
    original_square = from_square

    attack_bb = get_northeast_ray(attack_bb, from_square)
    attack_bb = get_southwest_ray(attack_bb, from_square)
    attack_bb = get_northwest_ray(attack_bb, from_square)
    attack_bb = get_southeast_ray(attack_bb, from_square)

    attack_bb = clear_bit(attack_bb, original_square)

    return attack_bb

##############################################################################################################
#                                                                                                            #
#   The Rook                                                                                                 #
#                                                                                                            #
##############################################################################################################

def make_rook_attack(bitboard: np.uint64) -> np.uint64:
            
    rook_attack = np.uint64(0)

    for square in get_rook_squares(bitboard):
        rook_attack |= gen_rook_attack_file(square) | gen_rook_attack_row(square)
    return rook_attack

def get_rook_squares(bitboard: np.uint64) -> int:
    
    rook_squares = []

    for square in range(64):
        if bitboard & np.uint64(1) << np.uint64(square):
            rook_squares.append(square)

    return rook_squares

def gen_rook_attack_row(square: int) -> np.uint64:
    """
    Generates rank attacks from the provided square on an otherwise empty bitboard
    :param square: starting square from which to generate rank attacks
    :return: np.uint64 rank attacks bitboard from the provided square
    """
    attack_bb = np.uint64(0)
    attack_bb = get_north_ray(attack_bb, square)
    attack_bb = get_south_ray(attack_bb, square)
    attack_bb = clear_bit(attack_bb, square)
    return attack_bb


def gen_rook_attack_file(square: int) -> np.uint64:
    """
    Generates file attacks from the provided square on an otherwise empty bitboard
    :param square: starting square from which to generate file attacks
    :return: np.uint64 file attacks bitboard from the provided square
    """
    attack_bb = np.uint64(0)
    attack_bb = get_east_ray(attack_bb, square)
    attack_bb = get_west_ray(attack_bb, square)
    attack_bb = clear_bit(attack_bb, square)
    return attack_bb

##############################################################################################################
#                                                                                                            #
#   The Queen                                                                                                #
#                                                                                                            #
##############################################################################################################

def make_queen_attack(bitboard: np.uint64) -> np.uint64:
                
    queen_attack = np.uint64(0)

    for square in get_queen_squares(bitboard):
        queen_attack |= gen_queen_attack(square)
    return queen_attack

def get_queen_squares(bitboard: np.uint64) -> int:
        
    queen_squares = []

    for square in range(64):
        if bitboard & np.uint64(1) << np.uint64(square):
            queen_squares.append(square)

    return queen_squares

def gen_queen_attack(square: int) -> np.uint64:

    return gen_bishop_attack(square) \
           | gen_rook_attack_row(square) \
           | gen_rook_attack_file(square)

##############################################################################################################
#                                                                                                            #
#   The King                                                                                                 #
#                                                                                                            #
##############################################################################################################

def make_king_attack(bitboard: np.uint64) -> np.uint64:
                        
    king_attack = np.uint64(0)

    for square in get_king_squares(bitboard):
        king_attack |= gen_king_attack(square)
    return king_attack

def get_king_squares(bitboard: np.uint64) -> int:
                
    king_squares = []

    for square in range(64):
        if bitboard & np.uint64(1) << np.uint64(square):
            king_squares.append(square)
    
    return king_squares

def gen_king_attack(square: int) -> np.uint64:

    attack_bb = np.uint64(0)
    for i in [8, -8]:
        # North-South
        attack_bb |= HOT << np.uint64(square + i)
    for i in [1, 9, -7]:
        # East (mask the A file)
        attack_bb |= HOT << np.uint64(square + i) & ~np.uint64(File.hexA)
    for i in [-1, -9, 7]:
        # West (mask the H file)
        attack_bb |= HOT << np.uint64(square + i) & ~np.uint64(File.hexH)
    return attack_bb

##############################################################################################################
#                                                                                                            #
#   Rays                                                                                                     #
#                                                                                                            #
##############################################################################################################

def get_south_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of south sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the south ray sliding attacks from `square`
    :param from_square: The square from a south-sliding piece attacks
    :return: np.uint64 bitboard of the southern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    for i in range(0, -64, -8):
        bitboard |= set_bit(bitboard, from_square + i)
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_north_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of north sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the north ray sliding attacks from `square`
    :param from_square: The square from a north-sliding piece attacks
    :return: np.uint64 bitboard of the northern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    for i in range(0, 64, 8):
        bitboard |= set_bit(bitboard, from_square + i)
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_west_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of west sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the west ray sliding attacks from `square`
    :param from_square: The square from a west-sliding piece attacks
    :return: np.uint64 bitboard of the western squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    if from_square % 8 == 0:
        bitboard |= HOT << np.uint64(from_square)
        from_square -= 1
    else:
        while not from_square % 8 == 0:
            bitboard |= HOT << np.uint64(from_square)
            from_square -= 1
        bitboard |= HOT << np.uint64(from_square)
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_east_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of east sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the east ray sliding attacks from `square`
    :param from_square: The square from a east-sliding piece attacks
    :return: np.uint64 bitboard of the eastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    if not from_square % 8:
        bitboard |= HOT << np.uint64(from_square)
        from_square += 1
    while not from_square % 8 == 0:
        bitboard |= HOT << np.uint64(from_square)
        from_square += 1
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_southeast_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of northeast sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northeast ray sliding attacks from `square`
    :param from_square: The square from a northeast-sliding piece attacks
    :return: np.uint64 bitboard of the northeastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    if from_square % 8 == 0 and from_square not in File.H:
        bitboard |= HOT << np.uint64(from_square)
        from_square -= 7
    while not from_square % 8 == 0 and from_square not in File.H:
        bitboard |= HOT << np.uint64(from_square)
        from_square -= 7
    bitboard |= HOT << np.uint64(from_square)
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_northwest_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of northwest sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northwest ray sliding attacks from `square`
    :param from_square: The square from a northwest-sliding piece attacks
    :return: np.uint64 bitboard of the northwestern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    if from_square % 8 == 0 and from_square not in File.A:
        bitboard |= HOT << np.uint64(from_square)
        from_square += 7
    while not from_square % 8 == 0 and from_square not in File.A:
        bitboard |= HOT << np.uint64(from_square)
        from_square += 7
    bitboard |= HOT << np.uint64(from_square)
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_southwest_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of southwest sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the southwest ray sliding attacks from `square`
    :param from_square: The square from a southwest-sliding piece attacks
    :return: np.uint64 bitboard of the southwestern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    if from_square % 8 == 0:
        bitboard |= HOT << np.uint64(from_square)
        from_square -= 9
    else:
        while not from_square % 8 == 0:
            bitboard |= HOT << np.uint64(from_square)
            from_square -= 9
        bitboard |= HOT << np.uint64(from_square)
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_northeast_ray(bitboard, from_square):
    """
    Returns a bitboard of northeast sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northeast ray sliding attacks from `square`
    :param from_square: The square from a northeast-sliding piece attacks
    :return: np.uint64 bitboard of the northeastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    if from_square % 8 == 0:
        bitboard |= HOT << np.uint64(from_square)
        from_square += 9
    while not from_square % 8 == 0:
        bitboard |= HOT << np.uint64(from_square)
        from_square += 9
    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard

