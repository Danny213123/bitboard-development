import chess_board


Test = chess_board.Board()
Test.update_board()
print(Test.white_pawn_bitboard)
print(Test.white_knight_attacks_bitboard)
print(Test.white_bishop_attacks)
print(Test.white_rook_attacks)
print(Test.white_queen_attacks)
print(Test.white_king_attacks)
print(Test.black_king_attacks)
print(Test.generate_fen())