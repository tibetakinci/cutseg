import os
import argparse
import shutil


def remove_superpixel_metadata_files(directory):
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    files_removed = 0
    for filename in os.listdir(directory):
        if filename.endswith('_superpixels.png') or filename.endswith('_metadata.csv'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                files_removed += 1
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

    print(f"Total files removed: {files_removed}")

def restructure_imgs_labels(directory):
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return
    
    imgs_moved, labels_moved = 0, 0
    imgs, labels = [], []
    for filename in os.listdir(directory):
        if filename.startswith('IMD'):
            img_path = os.path.join(directory, filename, f'{filename}_Dermoscopic_Image', f'{filename}.bmp')
            label_path = os.path.join(directory, filename, f'{filename}_lesion',  f'{filename}_lesion.bmp')
            
            imgs.append(img_path)
            labels.append(label_path)
    
    imgs_dir, labels_dir = os.path.join(directory, 'imgs'), os.path.join(directory, 'label')
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for img, label in zip(imgs, labels):
        img_dir = os.path.join(imgs_dir, img.split('/')[-1])
        label_dir = os.path.join(labels_dir, label.split('/')[-1])
        
        shutil.copyfile(img, img_dir)
        imgs_moved=imgs_moved+1
        
        shutil.copyfile(label, label_dir)
        labels_moved=labels_moved+1

    print(f"Total img files moved: {imgs_moved}; label files moved: {labels_moved}")


def prep_dataset(dataset, directory):
    if dataset.lower().startswith('isic'):
        remove_superpixel_metadata_files(directory)
    elif dataset.lower().startswith('ph2'):
        restructure_imgs_labels(directory)
    else:
        raise NotImplementedError(f'Preprocessing function for dataset {dataset} is not implemented.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Name of the dataset")
    parser.add_argument('--dir', type=str, help="The directory from which to remove files")
    args = parser.parse_args()
    
    prep_dataset(args.dataset, args.dir)

if __name__ == "__main__":
    main()
