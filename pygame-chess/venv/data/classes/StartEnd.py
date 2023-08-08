import pygame

class StartEnd:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 30)  # You can adjust the font and size

    def startDraw(self, display):
        start_text = self.font.render("Welcome! You will be shown a Mate-In-One Puzzle.", True, (255, 255, 255))  # White color
        start_rect = start_text.get_rect(center=(self.width // 2, (self.height // 2) - 50))
        display.blit(start_text, start_rect)
        start_text2 = self.font.render("You must solve this puzzle in 10 seconds.", True, (255, 255, 255))  # White color
        start_rect2 = start_text.get_rect(center=((self.width // 2) + 50, (self.height // 2) + 20))
        display.blit(start_text2, start_rect2)
        start_text2 = self.font.render("Good Luck!", True, (255, 255, 255))  # White color
        start_rect2 = start_text.get_rect(center=((self.width // 2) + 180, (self.height // 2) + 90))
        display.blit(start_text2, start_rect2)

    def endDraw(self, display, winner):
        end_text = self.font.render(f" {winner} Wins!", True, (255, 255, 255))  # White color
        end_rect = end_text.get_rect(center=(self.width // 2, self.height // 2))
        display.blit(end_text, end_rect)
    def timeUpDraw(self, display):
        end_text = self.font.render("Time's Up!", True, (255, 255, 255))  # White color
        end_rect = end_text.get_rect(center=(self.width // 2, self.height // 2))
        display.blit(end_text, end_rect)

