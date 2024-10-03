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
    screen.fill(BACKGROUND_COLOR)
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
    enemy_piece = 1 if piece == 2 else 2
    # north direction check
    if row - 2 >= 0:
        # normal piece capture
        if board[row - 1][col] == enemy_piece and (board[row - 2][col] == piece or (row - 2, col) in CORNERS):
            board[row - 1][col] = 0
    # north king capture
    if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
        if piece == 1 and board[row - 1][col] == 3 and (row - 2 < 0 or board[row - 2][col] == piece) and board[row - 1][col - 1] == piece and board[row - 1][col + 1] == piece:
            print("ATTACKERS WIN")
    # south direction check
    if row + 2 <= BOARD_SIZE - 1:
        if board[row + 1][col] == enemy_piece and (board[row + 2][col] == piece or (row + 2, col) in CORNERS):
            board[row + 1][col] = 0
    # south king capture
    if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
        if piece == 1 and board[row + 1][col] == 3 and (row + 2 < BOARD_SIZE or board[row + 2][col] == piece) and \
                board[row + 1][col - 1] == piece and board[row + 1][col + 1] == piece:
            print("ATTACKERS WIN")
    # west direction check
    if col - 2 >= 0:
        if board[row][col - 1] == enemy_piece and (board[row][col - 2] == piece or (row, col - 2) in CORNERS):
            board[row][col - 1] = 0
    # west king check
    if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
        if piece == 1 and board[row][col - 1] == 3 and (col - 2 < 0 or board[row][col - 2] == piece) and \
                board[row - 1][col - 1] == piece and board[row + 1][col - 1] == piece:
            print("ATTACKERS WIN")
    # east direction check
    if col + 2 <= BOARD_SIZE - 1:
        if board[row][col + 1] == enemy_piece and (board[row][col + 2] == piece or (row, col + 2) in CORNERS):
            board[row][col + 1] = 0
    # east king check
    if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
        if piece == 1 and board[row][col + 1] == 3 and (col + 2 > BOARD_SIZE - 1 or board[row][col + 2] == piece) and \
                board[row - 1][col + 1] == piece and board[row + 1][col + 1] == piece:
            print("ATTACKERS WIN")

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
        else:
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

"""   
Game loop

Piece movement code assisted by AI
"""
clock = pygame.time.Clock()
def main():
    setup_board()
    draw_board()

    global piece_arr
    global turn

    is_dragging = False
    selected_piece = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Check if we are clicking on a piece
                for piece in piece_arr:
                    if piece["rect"].collidepoint(mouse_x, mouse_y) and (turn == piece["piece"] or (piece["piece"] == 3 and turn == 2)):
                        is_dragging = True
                        selected_piece = piece
                        board[selected_piece["row"]][selected_piece["col"]] = 0
                        break

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
                        # play the piece
                        board[grid_y][grid_x] = piece
                        # if the piece is not going back to its og location, change turns
                        if not (grid_y == row and grid_x == col):
                            check_kills(grid_y, grid_x, piece)
                            turn = 1 if turn == 2 else 2
                    else:
                        # if the user is playing in a full spot, return the piece to its og location
                        board[row][col] = piece
                is_dragging = False

        piece_arr = []
        draw_board()

        if is_dragging:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.gfxdraw.filled_circle(
                screen,
                mouse_x,
                mouse_y,
                TILE_SIZE // 2 - 8,
                PIECE_COLORS[selected_piece["piece"]],
            )
            pygame.gfxdraw.aacircle(
                screen,
                mouse_x,
                mouse_y,
                TILE_SIZE // 2 - 8,
                BLACK
            )

        # Draw the piece
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()