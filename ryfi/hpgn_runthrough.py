import threading
import time

from pynput import keyboard

from game import Game

game = Game()

COL_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

moves_played = 0

class sim_game():
    def __init__(self):
        self.moves_played = 0
        thread = threading.Thread(target=lambda: self.start_listener())
        thread.start()
        game.play_game()

    # ai
    def on_press(self, key):
        try:
            # Check if the key pressed is Enter
            if key == keyboard.Key.right:
                self.run_file()

        except AttributeError:
            pass  # Handle other keys that might not have a 'char' attribute

    # ai
    def start_listener(self):
        # Create a listener for key press events
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def run_file(self):
        """
        Executes a series of moves read from a file, updating the game state incrementally.

        :return: None
        """
        global moves_played
        with open('saved_games/episode_50.hgpn', 'r') as file:
            moves_simmed = 0
            # Read the content of the file
            content = file.read().split("\n")

            game_content = content[4:]

            for row in game_content:
                moves = row.split(" ")
                for move in moves:
                    if move != "":
                        move = move.split("-")
                        old = str(move[0])[1:]
                        new = str(move[1])[1:]
                        if self.moves_played == moves_simmed:
                            if game.board[int(old[1:]) - 1][COL_NAMES.index(old[0])] == 3:
                                game.place_piece(COL_NAMES.index(new[0]), int(new[1:]) - 1, int(old[1:]) - 1,
                                             COL_NAMES.index(old[0]), 3)
                            else:
                                game.place_piece(COL_NAMES.index(new[0]), int(new[1:]) - 1, int(old[1:]) - 1,
                                                 COL_NAMES.index(old[0]), game.turn)
                            self.moves_played += 1
                            time.sleep(.1)
                            return
                        moves_simmed += 1


sim_game()




