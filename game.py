import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 11
TILE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE

# Color Constants
BACKGROUND_COLOR = (210, 79, 51)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ATTACKER_COLOR = BLACK
DEFENDER_COLOR = WHITE
KING_COLOR = (255, 255, 0)
PIECE_COLORS = [WHITE, ATTACKER_COLOR, DEFENDER_COLOR, KING_COLOR]

BOARD_START = [
    [0,0,0,1,1,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0],
    [1,0,0,0,0,2],
    [1,0,0,0,2,2],
    [1,1,0,2,2,3]
]
CORNERS = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
ADJUSTMENTS = [[5, 5, -6, -6], [0, 5, -5, -6], [5, 0, -6, -5], [0, 0, -5, -5]]

# Create the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Hnefatafl")

# Game Board
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# King position
king_position = (BOARD_SIZE // 2, BOARD_SIZE // 2)

# Place pieces (Attackers = 1, Defenders = 2, King = 3)
def setup_board():
    for row, vals in enumerate(BOARD_START):
        for col, val in enumerate(vals):
            board[row][col] = val
            board[-row-1][col] = val
            board[-row-1][-col-1] = val
            board[row][-col-1] = val

# Draw the board
def draw_board():
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, BLACK, (0, 0, WINDOW_SIZE, WINDOW_SIZE), 5)
    for corner, adjustment in zip(CORNERS, ADJUSTMENTS):
        row, col = corner
        pygame.draw.rect(screen, (170, 53, 28), (col * TILE_SIZE + adjustment[0], row * TILE_SIZE + adjustment[1], TILE_SIZE + adjustment[2], TILE_SIZE + adjustment[3]), 0)

    piece_arr = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pygame.draw.rect(screen, BLACK, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

            # Draw pieces
            if board[row][col] == 1:  # Attackers
                piece_arr.append(
                    [
                        pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                        1,
                        row,
                        col
                    ]
                )
                pygame.draw.circle(screen, PIECE_COLORS[1], (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2),
                                   TILE_SIZE // 2 - 5)
            elif board[row][col] == 2:  # Defenders
                piece_arr.append(
                    [
                        pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                        2,
                        row,
                        col
                    ]
                )
                pygame.draw.circle(screen, PIECE_COLORS[2], (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2),
                                   TILE_SIZE // 2 - 5)
            elif board[row][col] == 3:  # King
                piece_arr.append(
                    [
                        pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                        3,
                        row,
                        col
                    ]
                )
                pygame.draw.circle(screen, PIECE_COLORS[3],(col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2),TILE_SIZE // 2 - 5)
    return piece_arr

"""
pygame.draw.circle(screen, BLACK, (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2 - 5)
pygame.draw.circle(screen, WHITE, (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2 - 5)
pygame.draw.circle(screen, KING_COLOR, (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2 - 5)
"""
# Game loop
def main():
    setup_board()
    pieces = draw_board()
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
                for piece in pieces:
                    if piece[0].collidepoint(mouse_x, mouse_y):
                        is_dragging = True
                        selected_piece = piece
                        board[selected_piece[2]][selected_piece[3]] = 0
                        break

            elif event.type == pygame.MOUSEBUTTONUP:
                if is_dragging:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    grid_x = mouse_x // TILE_SIZE
                    grid_y = mouse_y // TILE_SIZE
                    board[grid_y][grid_x] = selected_piece[1]
                is_dragging = False

        pieces = draw_board()

        if is_dragging:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            selected_piece[0].center = (mouse_x, mouse_y)
            pygame.draw.circle(screen, PIECE_COLORS[selected_piece[1]], selected_piece[0].center, TILE_SIZE // 2 - 5)

        # Draw the piece
        pygame.display.flip()

if __name__ == "__main__":
    main()