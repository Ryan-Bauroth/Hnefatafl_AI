# class for saving games in hpgn format

GAME_NAME = "Hnefatafl"
GAME_RESULTS = ["0-0", "1-0", "0-1"]

class HPGN:
    def __init__(self, result, date, moves):
        # either 0 = tie, 1 = attackers, 2 = defenders
        self.result = GAME_RESULTS[result]
        # mm.dd.yyyy
        self.date = date
        # array of all moves in format a1-a2 or ka1-ka2
        self.moves = moves

    def create_file(self, filename):
        with open(filename + ".hgpn", 'w') as file:
            file.write(f"[Game \"{GAME_NAME}\"]\n")
            file.write(f"[Result \"{self.result}\"]\n")
            file.write(f"[Date \"{self.date}\"]\n")
            file.write("\n")

            turn_num = 1
            line_string = ""
            for move in self.moves:

                line_string += f"{move} "
                if turn_num % 50 == 0:
                    line_string = line_string.strip()
                    file.write(f"{line_string}\n")
                    line_string = ""
                turn_num += 1

