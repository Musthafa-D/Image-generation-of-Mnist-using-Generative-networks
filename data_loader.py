from ccbdl.data.utils.get_loader import get_loader
import matplotlib.pyplot as plt
import random

def prepare_data(data_config):
    loader = get_loader(data_config["dataset"])
    train_data, test_data, val_data = loader(**data_config).get_dataloader()
    
    view_data(train_data, data_config)
    view_data(test_data, data_config)

    return train_data, test_data, val_data

def view_data(data, data_config):
    # View the first image in train_data or test_data
    batch = next(iter(data))
    inputs, labels = batch

    # Set up the subplot dimensions
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    axs = axs.ravel()

    for i in range(10):
        idx = random.randint(0, data_config["batch_size"]-1)
        image = inputs[idx]
        label = labels[idx].item()  # Convert the label tensor to an integer

        image_np = image.permute(1, 2, 0).numpy()

        class_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                      6: "6", 7: "7", 8: "8", 9: "9"}

        # Display the image along with its label in the subplot
        axs[i].imshow(image_np, cmap='gray')
        axs[i].set_title(f"{class_dict[label]}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()