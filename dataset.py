import csv
import os

def adicionar_ao_dataset(image_name, peso, raca, idade):
    caminho_csv = 'datasets/metadata.csv'
    cabecalho = ['image', 'peso_real', 'raca', 'idade']
    
    arquivo_existe = os.path.exists(caminho_csv)

    with open(caminho_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=cabecalho)
        
        if not arquivo_existe:
            writer.writeheader()
            
        writer.writerow({
            'image': image_name,
            'peso_real': peso,
            'raca': raca,
            'idade': idade
        })

    label_path = f"datasets/labels/{image_name.split('.')[0]}.txt"
    if not os.path.exists(label_path):
        with open(label_path, 'w') as f:
            pass

    print(f"Registro de {image_name} finalizado.")

if __name__ == "__main__":
    if not os.path.exists('datasets/images'):
        os.makedirs('datasets/images')
    if not os.path.exists('datasets/labels'):
        os.makedirs('datasets/labels')

    adicionar_ao_dataset("img_001.jpg", 420, "Nelore", "24 meses")
    adicionar_ao_dataset("img_002.jpg", 390, "Angus", "18 meses")