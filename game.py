class Game:
    def __init__(self, *, galaxy, turn=0):
        self.current_turn = turn

    def process_turn(self):
        # Do stuff
        self.turn += 1
