# menu_manager.py
import pygame

class Button:
    def __init__(self, image_path, x_ratio, y_ratio, screen_size, scale=1.0):
        self.original_image = pygame.image.load(image_path).convert_alpha()
        # Initial scaling based on given scale factor
        self.image = pygame.transform.scale(self.original_image, (int(self.original_image.get_width() * scale), int(self.original_image.get_height() * scale)))
        self.x_ratio = x_ratio
        self.y_ratio = y_ratio
        self.screen_size = screen_size
        self.update_position()

    def update_position(self):
        # Calculate absolute position based on ratios and current screen size
        screen_width, screen_height = self.screen_size
        self.rect = self.image.get_rect(center=(int(screen_width * self.x_ratio), int(screen_height * self.y_ratio)))

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

def show_main_menu(screen):
    screen_width, screen_height = screen.get_size()

    # Title Image
    title_img_raw = pygame.image.load('assets/BUTTON TITLE/Title.png').convert_alpha()
    title_width_scaled = int(screen_width * 0.8)
    title_height_scaled = int(title_img_raw.get_height() * (title_width_scaled / title_img_raw.get_width()))
    title_img = pygame.transform.scale(title_img_raw, (title_width_scaled, title_height_scaled))
    title_rect = title_img.get_rect(center=(screen_width // 2, int(screen_height * 0.3)))

    # Play Button
    play_button = Button('assets/BUTTON TITLE/PLAY.png', 0.5, 0.65, (screen_width, screen_height), scale=0.8)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if play_button.is_clicked(event):
                return True # Play button clicked, proceed to game

        screen.fill((0, 0, 0)) # Black background
        screen.blit(title_img, title_rect)
        play_button.draw(screen)

        pygame.display.flip()
    return False

def run_menu():
    pygame.init()
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Two Player Blink Shot")

    if show_main_menu(screen): # If main menu proceeds
        pygame.quit() # Quit Pygame before starting CV2 game
        return True # Signal to start the main game
    pygame.quit()
    return False # Game not started