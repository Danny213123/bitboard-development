import numpy as np
import chess

from chess_bitboard_helpers import print_bitboard, print_entire_board, bitboard_to_array, bitboards_to_array, \
    make_knight_attack, gen_knight_attack, get_knight_squares, make_bishop_attack, gen_bishop_attack, \
    get_bishop_squares, make_rook_attack, get_rook_squares, make_queen_attack, get_queen_squares, \
    make_king_attack, get_king_squares

from chess_constants import Piece, Colour, File

class Board:

    def __init__(self):

        self.board = chess.Board()

    def generate_fen(self):

        return self.board.fen()
    
    def update_board(self):

        black, white = self.board.occupied_co

        bit_boards = np.array([
            black & self.board.pawns,
            black & self.board.rooks,
            black & self.board.knights,
            black & self.board.bishops,
            black & self.board.queens,
            black & self.board.kings,
            white & self.board.pawns,
            white & self.board.rooks,
            white & self.board.knights,
            white & self.board.bishops,
            white & self.board.queens,
            white & self.board.kings,
        ], dtype=np.uint64)

        self.black_pawn_bitboard   = bit_boards[0]
        self.black_rook_bitboard   = bit_boards[1]
        self.black_knight_bitboard = bit_boards[2]
        self.black_bishop_bitboard = bit_boards[3]
        self.black_queen_bitboard  = bit_boards[4]
        self.black_king_bitboard   = bit_boards[5]

        self.white_pawn_bitboard   = bit_boards[6]
        self.white_rook_bitboard   = bit_boards[7]
        self.white_knight_bitboard = bit_boards[8]
        self.white_bishop_bitboard = bit_boards[9]
        self.white_queen_bitboard  = bit_boards[10]
        self.white_king_bitboard   = bit_boards[11]

        #self.knight_locations = get_knight_squares(self.white_knight_bitboard | self.black_knight_bitboard)

        self.white_knight_attacks_bitboard = self.white_knight_attacks
        self.black_knight_attacks_bitboard = self.black_knight_attacks

        self.white_bishop_attacks_bitboard = self.white_bishop_attacks
        self.black_bishop_attacks_bitboard = self.black_bishop_attacks

        self.white_rook_attacks_bitboard = self.white_rook_attacks
        self.black_rook_attacks_bitboard = self.black_rook_attacks

        self.white_queen_attacks_bitboard = self.white_queen_attacks
        self.black_queen_attacks_bitboard = self.black_queen_attacks

        self.white_king_attacks_bitboard = self.white_king_attacks
        self.black_king_attacks_bitboard = self.black_king_attacks
    
    ##############################################################################################################
    #                                                                                                            #
    #   Properties                                                                                               #
    #                                                                                                            #
    ##############################################################################################################

    @property
    def get_board(self):
        return self.Board
    
    @property
    def white_pieces(self):
        return self.white_pawn_bitboard | self.white_knight_bitboard | self.white_bishop_bitboard | self.white_rook_bitboard | self.white_queen_bitboard | self.white_king_bitboard
    
    @property
    def black_pieces(self):
        return self.black_pawn_bitboard | self.black_knight_bitboard | self.black_bishop_bitboard | self.black_rook_bitboard | self.black_queen_bitboard | self.black_king_bitboard
    
    @property
    def empty_squares(self):
        return ~(self.white_pieces | self.black_pieces)
    
    @property
    def occupied_squares(self):
        return self.white_pieces | self.black_pieces
    
    @property
    def white_pawn_east_attacks(self):
        return (self.white_pawn_bitboard << np.uint64(9)) & ~np.uint64(File.hexA)
    
    @property
    def white_pawn_west_attacks(self):
        return (self.white_pawn_bitboard << np.uint64(7)) & ~np.uint64(File.hexH)
    
    @property
    def white_pawn_attacks(self):
        return self.white_pawn_east_attacks | self.white_pawn_west_attacks
    
    @property
    def black_pawn_east_attacks(self):
        return (self.black_pawn_bitboard >> np.uint64(7)) & ~np.uint64(File.hexA)
    
    @property
    def black_pawn_west_attacks(self):
        return (self.black_pawn_bitboard >> np.uint64(9)) & ~np.uint64(File.hexH)
    
    @property
    def black_pawn_attacks(self):
        return self.black_pawn_east_attacks | self.black_pawn_west_attacks
    
    @property
    def white_knight_attacks(self,):
        return make_knight_attack(self.white_knight_bitboard)
    
    @property
    def black_knight_attacks(self):
        return make_knight_attack(self.black_knight_bitboard)
    
    @property
    def white_bishop_attacks(self):
        return make_bishop_attack(self.white_bishop_bitboard)
    
    @property
    def black_bishop_attacks(self):
        return make_bishop_attack(self.black_bishop_bitboard)
    
    @property
    def white_rook_attacks(self):
        return make_rook_attack(self.white_rook_bitboard)
    
    @property
    def black_rook_attacks(self):
        return make_rook_attack(self.black_rook_bitboard)
    
    @property
    def white_queen_attacks(self):
        return make_queen_attack(self.white_queen_bitboard)
    
    @property
    def black_queen_attacks(self):
        return make_queen_attack(self.black_queen_bitboard)
    
    @property
    def white_king_attacks(self):
        return make_king_attack(self.white_king_bitboard)
    
    @property
    def black_king_attacks(self):
        return make_king_attack(self.black_king_bitboard)
    