

class Game:
    def __init__(self, board):
        self.board = board
 
    def one_move(self):

        def count_live_neighbors(board, row, column):
            Live_neighbors = 0
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
            for direction in directions:
                neighbor_row = row + direction[0]
                neighbor_column = column + direction[1]
                if neighbor_column < 0 or neighbor_column >= len(board[0]) or neighbor_row < 0 or neighbor_row >= len(board):
                    continue
                if board[neighbor_row][neighbor_column] == 1:
                    Live_neighbors += 1
            return Live_neighbors

        board = self.board

        new_board = []
        for i in range(len(board)):
            row = []
            for j in range(len(board[0])):
                Live_neighbors_num = count_live_neighbors(board, i, j)
                if board[i][j] == 1:
                    if Live_neighbors_num < 2:
                        row.append(0)
                    elif Live_neighbors_num <= 3:
                        row.append(1)
                    else:
                        row.append(0)
                else:
                    if Live_neighbors_num == 3:
                        row.append(1)
                    else:
                        row.append(0)
            
            new_board.append(row)
        
        self.board = new_board

if __name__ == "__main__":
    board = [[0,1,0,0,1],
             [1,0,0,0,0],
             [1,0,0,0,1],
             [1,1,1,1,0]]
    game = Game(board)
    game.one_move()
    print(game.board)


            


