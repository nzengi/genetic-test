import random
import hashlib
import numpy as np
from sklearn.neural_network import MLPRegressor
import ecdsa

# Hedef Hash değeri
target_ripemd160 = "20d45a6a762535700ce9e0b216e31994335db8a5"

# 64 haneli hexadecimal private key aralığı
start_range = int('00000000000000000000000000000000000000000000000000000000000000001', 16)
end_range = int('000000000000000000000000000000000000000000000000fffffffffffffffffff', 16)

# Geçerli bir private key olup olmadığını kontrol eden fonksiyon
def is_valid_private_key(private_key):
    try:
        ecdsa.SigningKey.from_string(private_key.to_bytes(32, byteorder='big'), curve=ecdsa.SECP256k1)
        return True
    except ValueError:
        return False

# Private key'den public key üretme
def private_key_to_public_key(private_key):
    sk = ecdsa.SigningKey.from_string(private_key.to_bytes(32, byteorder='big'), curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    public_key = b'\x04' + vk.to_string()
    return public_key

# Public key'i compress etme
def compress_public_key(public_key):
    if public_key[-1] % 2 == 0:
        return b'\x02' + public_key[1:33]
    else:
        return b'\x03' + public_key[1:33]

# RIPEMD-160 Hash oluşturma
def hash160(public_key):
    sha256 = hashlib.sha256(public_key).digest()
    ripemd160 = hashlib.new('ripemd160', sha256).digest()
    return ripemd160

# Rastgele 64 haneli private key oluşturma
def create_initial_population(pop_size, start, end):
    population = []
    for _ in range(pop_size):
        while True:
            full_key = random.randint(start, end)
            if is_valid_private_key(full_key):
                population.append(full_key)
                break
    return population

# Genetik operatör: Crossover
def crossover(parent1, parent2):
    parent1_bytes = parent1.to_bytes(32, byteorder='big')
    parent2_bytes = parent2.to_bytes(32, byteorder='big')

    crossover_point = random.randint(1, len(parent1_bytes) - 1)

    child1_bytes = parent1_bytes[:crossover_point] + parent2_bytes[crossover_point:]
    child2_bytes = parent2_bytes[:crossover_point] + parent1_bytes[crossover_point:]

    child1 = int.from_bytes(child1_bytes, byteorder='big')
    child2 = int.from_bytes(child2_bytes, byteorder='big')

    return child1, child2

# Mutasyon fonksiyonu
def mutate(individual, mutation_rate=0.01):
    individual_list = list(individual.to_bytes(32, byteorder='big'))
    for i in range(len(individual_list)):
        if random.random() < mutation_rate:
            individual_list[i] = random.choice(range(256))
    return int.from_bytes(bytes(individual_list), byteorder='big')

# Kuantum esinlenmeli optimizasyon
def quantum_tunneling_optimization(individual, search_space):
    individual_bytes = individual.to_bytes(32, byteorder='big')

    random_jump = random.randint(1, len(individual_bytes) - 1)

    new_individual_bytes = individual_bytes[:random_jump] + random.choice([x.to_bytes(32, byteorder='big') for x in search_space])[random_jump:]
    new_individual = int.from_bytes(new_individual_bytes, byteorder='big')

    return new_individual

# Fitness fonksiyonu
def fitness_function(private_key, target_ripemd160):
    public_key = private_key_to_public_key(private_key)
    compressed_pub_key = compress_public_key(public_key)
    ripemd160_hash = hash160(compressed_pub_key).hex()
    return abs(int(ripemd160_hash, 16) - int(target_ripemd160, 16))

# Yapay sinir ağı modelini eğitme
def train_model(population, fitness_scores):
    X = np.array([list(private_key.to_bytes(32, byteorder='big')) for private_key in population])
    y = np.array(fitness_scores)
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(X, y)
    return model

# Makine öğrenimi ile tahmin fonksiyonu
def predict_fitness(model, private_key):
    key_features = np.array(list(private_key.to_bytes(32, byteorder='big'))).reshape(1, -1)
    return model.predict(key_features)[0]

# Ana fonksiyon
if __name__ == "__main__":
    # Başlangıç popülasyonunu oluştur
    population = create_initial_population(100, start_range, end_range)

    # Fitness değerlerini hesapla
    fitness_scores = [fitness_function(key, target_ripemd160) for key in population]

    # Yapay sinir ağı modelini eğit
    model = train_model(population, fitness_scores)

    # Evrimsel süreç, Best Fitness = 0'a ulaşana kadar çalışacak
    generation = 0
    while True:
        new_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        # Yeni popülasyonu değerlendir
        population = sorted(new_population, key=lambda x: predict_fitness(model, x))[:len(population)]

        # Kuantum optimizasyonu uygula
        population = [quantum_tunneling_optimization(individual, population) for individual in population]

        # En iyi sonuçları yazdır
        best_individual = population[0]
        best_fitness = fitness_function(best_individual, target_ripemd160)
        print(f'Generation {generation}: Best fitness = {best_fitness}')

        # Fitness değeri 0'a ulaştığında döngüyü sonlandır
        if best_fitness == 0:
            print(f'Target private key found: {best_individual}')
            break

        # Makine öğrenimi modelini yeniden eğit
        fitness_scores = [fitness_function(key, target_ripemd160) for key in population]
        model = train_model(population, fitness_scores)

        generation += 1
