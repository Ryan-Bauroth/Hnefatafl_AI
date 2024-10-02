import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 11
TILE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ATTACKER_COLOR = (255, 0, 0)
DEFENDER_COLOR = (0, 0, 255)
KING_COLOR = (255, 255, 0)

# Create the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Hnefatafl")

# Game Board
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# King position
king_position = (BOARD_SIZE // 2, BOARD_SIZE // 2)

# Place pieces (Attackers = 1, Defenders = 2, King = 3)
def setup_board():
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row == 0 or row == BOARD_SIZE-1) and 3 <= col <= BOARD_SIZE - 4:
                board[row][col] = 1  # Attackers
            elif (col == 0 or col == BOARD_SIZE-1) and 3 <= row <= BOARD_SIZE - 4:
                board[row][col] = 1  # Attackers
            elif (row in [3, 4, 6, 7] and col == 5) or (col in [3, 4, 6, 7] and row == 5):
                board[row][col] = 2  # Defenders
    board[king_position[0]][king_position[1]] = 3  # King

# Draw the board
def draw_board():
    screen.fill(WHITE)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pygame.draw.rect(screen, BLACK, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

            # Draw pieces
            if board[row][col] == 1:  # Attackers
                pygame.draw.circle(screen, ATTACKER_COLOR, 
                                   (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2 - 5)
            elif board[row][col] == 2:  # Defenders
                pygame.draw.circle(screen, DEFENDER_COLOR, 
                                   (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2 - 5)
            elif board[row][col] == 3:  # King
                pygame.draw.circle(screen, KING_COLOR, 
                                   (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2 - 5)

# Game loop
def main():
    setup_board()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_board()
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()