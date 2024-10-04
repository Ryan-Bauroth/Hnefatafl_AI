import pygame
import sys
from pygame import gfxdraw

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 11
TILE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE
FPS = 60

# Color Constants
BACKGROUND_COLOR = (247,247,247)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (180, 180, 180)
LATEST_MOVE_COLOR = (206, 223, 159)
KILL_COLOR = (245, 232, 175)
KING_COLOR = (255, 255, 0)
CORNER_COLOR = (233, 196, 177)
PIECE_COLORS = [(0, 0, 0), BLACK, WHITE, KING_COLOR]

# Board Constants
BOARD_START = [
    [0,0,0,1,1,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0],
    [1,0,0,0,0,2],
    [1,0,0,0,2,2],
    [1,1,0,2,2,3]
]
CORNERS = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
CENTER = (BOARD_SIZE // 2, BOARD_SIZE // 2)

# Load piece images and scale them to fit the tile size
attacker_img = pygame.image.load('pieces/attacker.png')
attacker_img = pygame.transform.smoothscale(attacker_img, (TILE_SIZE - 10, TILE_SIZE - 10))

defender_img = pygame.image.load('pieces/defender.png')
defender_img = pygame.transform.smoothscale(defender_img, (TILE_SIZE - 10, TILE_SIZE - 10))

king_img = pygame.image.load('pieces/king.png')
king_img = pygame.transform.smoothscale(king_img, (TILE_SIZE - 10, TILE_SIZE - 10))

# Create the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Hnefatafl")

# Game Vars
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
recent_move_cords = {}
piece_arr = []
turn = 1

# Set up board by mirroring 1/4th board (Attackers = 1, Defenders = 2, King = 3)
def setup_board():
    for row, vals in enumerate(BOARD_START):
        for col, val in enumerate(vals):
            board[row][col] = val
            board[-row-1][col] = val
            board[-row-1][-col-1] = val
            board[row][-col-1] = val

# Updates piece arr and draws piece on the screen
def append_draw_piece(row, col, piece):
    piece_arr.append({
        "rect": pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
        "piece": piece,
        "row": row,
        "col": col,
    })
    pygame.gfxdraw.filled_circle(
        screen,
        col * TILE_SIZE + TILE_SIZE // 2,
        row * TILE_SIZE + TILE_SIZE // 2,
        TILE_SIZE // 2 - 9,
        PIECE_COLORS[piece],
    )
    pygame.gfxdraw.aacircle(
        screen,
        col * TILE_SIZE + TILE_SIZE // 2,
        row * TILE_SIZE + TILE_SIZE // 2,
        TILE_SIZE // 2 - 9,
        BLACK
    )


    # piece_x = col * TILE_SIZE + TILE_SIZE // 2
    # piece_y = row * TILE_SIZE + TILE_SIZE // 2
    #
    # if piece == 1:  # Attacker
    #     screen.blit(attacker_img, (piece_x - TILE_SIZE // 2 + 5, piece_y - TILE_SIZE // 2 + 5))
    # elif piece == 2:  # Defender
    #     screen.blit(defender_img, (piece_x - TILE_SIZE // 2 + 5, piece_y - TILE_SIZE // 2 + 5))
    # elif piece == 3:  # King
    #     screen.blit(king_img, (piece_x - TILE_SIZE // 2 + 5, piece_y - TILE_SIZE // 2 + 5))

# Draw the board
def draw_board():
    pygame.draw.rect(screen, LIGHT_GRAY, (0, 0, WINDOW_SIZE, WINDOW_SIZE), 2)
    for corner in CORNERS:
        row, col = corner
        pygame.draw.rect(screen, CORNER_COLOR, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)
    c_row, c_col = CENTER
    pygame.draw.rect(screen, CORNER_COLOR, (c_col * TILE_SIZE, c_row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pygame.draw.rect(screen, LIGHT_GRAY, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

            # Draw pieces
            if board[row][col] == 1:  # Attackers
                append_draw_piece(row, col, 1)
            elif board[row][col] == 2:  # Defenders
                append_draw_piece(row, col, 2)
            elif board[row][col] == 3:  # King
                append_draw_piece(row, col, 3)

"""
Checks for kills

:param row: int the most recently moved piece's row
:param col: int the most recently moved piece's col
:param piece: int the most recently moved piece's piece id
"""
def check_kills(row, col, piece):
    kill_cords = []
    enemy_piece = 1 if piece == 2 else 2

    # north direction check
    if row - 2 >= 0:
        # normal piece capture
        if board[row - 1][col] == enemy_piece and (board[row - 2][col] == piece or (row - 2, col) in CORNERS):
            board[row - 1][col] = 0
            kill_cords.append((row - 1, col))

    # north king capture
    if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
        if piece == 1 and board[row - 1][col] == 3 and (row - 2 < 0 or board[row - 2][col] == piece) and board[row - 1][col - 1] == piece and board[row - 1][col + 1] == piece:
            print("ATTACKERS WIN")

    # south direction check
    if row + 2 <= BOARD_SIZE - 1:
        if board[row + 1][col] == enemy_piece and (board[row + 2][col] == piece or (row + 2, col) in CORNERS):
            board[row + 1][col] = 0
            kill_cords.append((row + 1, col))

    # south king capture
    if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
        if piece == 1 and board[row + 1][col] == 3 and (row + 2 < BOARD_SIZE or board[row + 2][col] == piece) and \
                board[row + 1][col - 1] == piece and board[row + 1][col + 1] == piece:
            print("ATTACKERS WIN")

    # west direction check
    if col - 2 >= 0:
        if board[row][col - 1] == enemy_piece and (board[row][col - 2] == piece or (row, col - 2) in CORNERS):
            board[row][col - 1] = 0
            kill_cords.append((row, col - 1))

    # west king check
    if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
        if piece == 1 and board[row][col - 1] == 3 and (col - 2 < 0 or board[row][col - 2] == piece) and \
                board[row - 1][col - 1] == piece and board[row + 1][col - 1] == piece:
            print("ATTACKERS WIN")

    # east direction check
    if col + 2 <= BOARD_SIZE - 1:
        if board[row][col + 1] == enemy_piece and (board[row][col + 2] == piece or (row, col + 2) in CORNERS):
            board[row][col + 1] = 0
            kill_cords.append((row, col + 1))

    # east king check
    if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
        if piece == 1 and board[row][col + 1] == 3 and (col + 2 > BOARD_SIZE - 1 or board[row][col + 2] == piece) and \
                board[row - 1][col + 1] == piece and board[row + 1][col + 1] == piece:
            print("ATTACKERS WIN")

    return kill_cords

"""
Checks to see if an intended play is legal

:param grid_y: int the intended play's row
:param row: int the moving pieces original row
:param grid_x: int the intended play's col
:param col: int the moving pieces original col
"""
def check_play(grid_y, row, grid_x, col, piece):
    # pieces move like the rook in chess
    if not (grid_y == row or grid_x == col):
        return False
    # pieces cannot move onto the corners (except for kings)
    if (grid_y, grid_x) in CORNERS or (grid_y, grid_x) == CENTER:
        if piece != 3:
            return False
        elif (grid_y, grid_x) in CORNERS:
            print("DEFENDERS WIN")
            return True
    # thus moving in the x direction
    if grid_y == row:
        # if moving upward
        if grid_x - col > 0:
            for c in range(col + 1, grid_x):
                if board[row][c] != 0:
                    return False
        # if moving downward
        if grid_x - col < 0:
            for c in range(col - 1, grid_x, -1):
                if board[row][c] != 0:
                    return False
    # thus moving in the y direction
    else:
        # if moving right
        if grid_y - row > 0:
            for r in range(row + 1, grid_y):
                if board[r][col] != 0:
                    return False
        # if moving left
        if grid_y - row < 0:
            for r in range(row - 1, grid_y, -1):
                if board[r][col] != 0:
                    return False
    return True

def update_field_colors(col, row, color):
    pygame.draw.rect(screen, color, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)

def clear_field_colors():
    global recent_move_cords
    recent_move_cords = {
        "original_col": -1,
        "original_row": -1,
        "new_col": -1,
        "new_row": -1,
    }

def update_possible_moves(col, row, piece):
    global possible_moves
    possible_moves = []
    for r in range(row - 1, -1, -1):
        if board[r][col] == 0:
            if piece == 3 or ((r, col) not in CORNERS and (r, col) != CENTER):
                possible_moves.append((r, col))
        else:
            break
    for r in range(row + 1, BOARD_SIZE):
        if board[r][col] == 0:
            if piece == 3 or ((r, col) not in CORNERS and (r, col) != CENTER):
                possible_moves.append((r, col))
        else:
            break
    for c in range(col - 1, -1, -1):
        if board[row][c] == 0:
            if piece == 3 or ((row, c) not in CORNERS and (row, c) != CENTER):
                possible_moves.append((row, c))
        else:
            break
    for c in range(col + 1, BOARD_SIZE):
        if board[row][c] == 0:
            if piece == 3 or ((row, c) not in CORNERS and (row, c) != CENTER):
                possible_moves.append((row, c))
        else:
            break





def draw_possible_moves():
    global possible_moves
    for move in possible_moves:
        row, col = move
        pygame.gfxdraw.filled_circle(
            screen,
            col * TILE_SIZE + TILE_SIZE // 2,
            row * TILE_SIZE + TILE_SIZE // 2,
            TILE_SIZE // 2 - 20,
            KILL_COLOR,
        )
        pygame.gfxdraw.aacircle(
            screen,
            col * TILE_SIZE + TILE_SIZE // 2,
            row * TILE_SIZE + TILE_SIZE // 2,
            TILE_SIZE // 2 - 20,
            LIGHT_GRAY,
        )

def place_piece(grid_x, grid_y, row, col, piece):
    global turn
    global kill_cords
    global recent_move_cords
    global possible_moves
    # play the piece
    board[grid_y][grid_x] = piece
    # if the piece is not going back to its og location, change turns
    if not (grid_y == row and grid_x == col):
        board[row][col] = 0
        kill_cords = check_kills(grid_y, grid_x, piece)
        recent_move_cords = {
            "original_col": col,
            "original_row": row,
            "new_col": grid_x,
            "new_row": grid_y,
        }
        possible_moves = []
        turn = 1 if turn == 2 else 2

kill_cords = []
possible_moves = []

"""   
Game loop

Piece movement code assisted by AI
"""
clock = pygame.time.Clock()
def main():
    setup_board()
    draw_board()
    clear_field_colors()

    global piece_arr
    global turn
    global recent_move_cords
    global kill_cords
    global possible_moves

    is_dragging = False
    selected_piece = None
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                piece_clicked = False

                mouse_x, mouse_y = event.pos

                # Check if we are clicking on a piece
                for piece in piece_arr:
                    if piece["rect"].collidepoint(mouse_x, mouse_y) and (turn == piece["piece"] or (piece["piece"] == 3 and turn == 2)):
                        update_possible_moves(piece["col"], piece["row"], piece["piece"])
                        is_dragging = True
                        piece_clicked = True
                        selected_piece = piece
                        board[selected_piece["row"]][selected_piece["col"]] = 0
                        break
                if not piece_clicked:
                    if selected_piece is not None:
                        grid_x = mouse_x // TILE_SIZE
                        grid_y = mouse_y // TILE_SIZE
                        row = selected_piece["row"]
                        col = selected_piece["col"]
                        piece = selected_piece["piece"]
                        if (grid_y, grid_x) in possible_moves:
                            if board[grid_y][grid_x] == 0 and check_play(grid_y, row, grid_x, col, piece):
                                place_piece(grid_x, grid_y, row, col, piece)
                    possible_moves = []



            elif event.type == pygame.MOUSEBUTTONUP:
                if is_dragging:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = selected_piece["row"]
                    col = selected_piece["col"]
                    piece = selected_piece["piece"]
                    grid_x = mouse_x // TILE_SIZE
                    grid_y = mouse_y // TILE_SIZE

                    # If the user is playing in a empty spot
                    if board[grid_y][grid_x] == 0 and check_play(grid_y, row, grid_x, col, piece):
                        place_piece(grid_x, grid_y, row, col, piece)
                    else:
                        board[row][col] = piece
                is_dragging = False

        piece_arr = []
        screen.fill(BACKGROUND_COLOR)
        if recent_move_cords["original_row"] != -1:
            update_field_colors(recent_move_cords["original_col"], recent_move_cords["original_row"], LATEST_MOVE_COLOR)
            update_field_colors(recent_move_cords["new_col"], recent_move_cords["new_row"], LATEST_MOVE_COLOR)
        for kill_cord in kill_cords:
            x, y = kill_cord
            update_field_colors(y, x, KILL_COLOR)
        draw_board()
        draw_possible_moves()

        if is_dragging:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.gfxdraw.filled_circle(
                screen,
                mouse_x,
                mouse_y,
                TILE_SIZE // 2 - 9,
                PIECE_COLORS[selected_piece["piece"]],
            )
            pygame.gfxdraw.aacircle(
                screen,
                mouse_x,
                mouse_y,
                TILE_SIZE // 2 - 9,
                BLACK
            )

        # Draw the piece
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()