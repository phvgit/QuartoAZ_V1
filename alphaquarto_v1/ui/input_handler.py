# -*- coding: utf-8 -*-
"""Gestion des entrees utilisateur"""


class InputHandler:
    """Gere les entrees utilisateur"""
    
    @staticmethod
    def get_square_input(legal_moves):
        """Demande un carre valide"""
        while True:
            try:
                square = int(input("Choisissez un carre (0-15): "))
                if square in legal_moves:
                    return square
                print("Carre non valide ou deja occupe!")
            except ValueError:
                print("Entree invalide!")
            except KeyboardInterrupt:
                print("\nJeu interrompu")
                exit()
    
    @staticmethod
    def get_piece_input(available_pieces):
        """Demande une piece valide"""
        while True:
            try:
                print(f"Pieces disponibles: {available_pieces}")
                piece = int(input("Choisissez une piece: "))
                if piece in available_pieces:
                    return piece
                print("Piece non disponible!")
            except ValueError:
                print("Entree invalide!")
            except KeyboardInterrupt:
                print("\nJeu interrompu")
                exit()
    
    @staticmethod
    def get_yes_no(prompt):
        """Demande oui ou non"""
        while True:
            response = input(f"{prompt} (o/n): ").lower()
            if response in ['o', 'oui', 'yes', 'y']:
                return True
            elif response in ['n', 'non', 'no']:
                return False
            else:
                print("Reponse invalide!")
