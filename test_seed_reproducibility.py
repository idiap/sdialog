#!/usr/bin/env python3
"""
Script de test pour démontrer la reproductibilité avec les graines dans BasicRoomGenerator
"""

import sys
import os

# Ajouter le chemin du module sdialog
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sdialog.audio.room_generator import BasicRoomGenerator

def test_reproducibility():
    """
    Teste la reproductibilité avec différentes graines
    """
    print("=== Test de reproductibilité avec BasicRoomGenerator ===\n")
    
    # Test 1: Sans graine (aléatoire)
    print("1. Génération sans graine (aléatoire):")
    generator_no_seed = BasicRoomGenerator()
    
    for i in range(3):
        room = generator_no_seed.generate({"room_size": 20})
        print(f"   Génération {i+1}: {room.dimensions.width:.2f}x{room.dimensions.length:.2f}x{room.dimensions.height:.2f}m, "
              f"ratio={room.aspect_ratio}, reverb={room.reverberation_time_ratio:.3f}")
    
    print()
    
    # Test 2: Avec graine fixe (reproductible)
    print("2. Génération avec graine fixe (seed=42):")
    generator_seed_42 = BasicRoomGenerator(seed=42)
    
    for i in range(3):
        room = generator_seed_42.generate({"room_size": 20})
        print(f"   Génération {i+1}: {room.dimensions.width:.2f}x{room.dimensions.length:.2f}x{room.dimensions.height:.2f}m, "
              f"ratio={room.aspect_ratio}, reverb={room.reverberation_time_ratio:.3f}")
    
    print()
    
    # Test 3: Vérification de la reproductibilité
    print("3. Vérification de la reproductibilité (même graine = même résultat):")
    
    # Première série avec seed=123
    generator1 = BasicRoomGenerator(seed=123)
    room1_1 = generator1.generate({"room_size": 15})
    room1_2 = generator1.generate({"room_size": 15})
    
    # Deuxième série avec la même graine
    generator2 = BasicRoomGenerator(seed=123)
    room2_1 = generator2.generate({"room_size": 15})
    room2_2 = generator2.generate({"room_size": 15})
    
    print(f"   Première série (seed=123):")
    print(f"     Room 1: {room1_1.dimensions.width:.2f}x{room1_1.dimensions.length:.2f}x{room1_1.dimensions.height:.2f}m, "
          f"ratio={room1_1.aspect_ratio}, reverb={room1_1.reverberation_time_ratio:.3f}")
    print(f"     Room 2: {room1_2.dimensions.width:.2f}x{room1_2.dimensions.length:.2f}x{room1_2.dimensions.height:.2f}m, "
          f"ratio={room1_2.aspect_ratio}, reverb={room1_2.reverberation_time_ratio:.3f}")
    
    print(f"   Deuxième série (seed=123):")
    print(f"     Room 1: {room2_1.dimensions.width:.2f}x{room2_1.dimensions.length:.2f}x{room2_1.dimensions.height:.2f}m, "
          f"ratio={room2_1.aspect_ratio}, reverb={room2_1.reverberation_time_ratio:.3f}")
    print(f"     Room 2: {room2_2.dimensions.width:.2f}x{room2_2.dimensions.length:.2f}x{room2_2.dimensions.height:.2f}m, "
          f"ratio={room2_2.aspect_ratio}, reverb={room2_2.reverberation_time_ratio:.3f}")
    
    # Vérification
    rooms_identical = (
        room1_1.dimensions.width == room2_1.dimensions.width and
        room1_1.dimensions.length == room2_1.dimensions.length and
        room1_1.dimensions.height == room2_1.dimensions.height and
        room1_1.aspect_ratio == room2_1.aspect_ratio and
        abs(room1_1.reverberation_time_ratio - room2_1.reverberation_time_ratio) < 1e-10
    )
    
    print(f"\n   ✅ Reproductibilité: {'RÉUSSIE' if rooms_identical else 'ÉCHOUÉE'}")
    
    print("\n=== Test terminé ===")

if __name__ == "__main__":
    test_reproducibility()
