from classification.model import *


def main():
    model_name = "DenseNext32"
    num_classes = 10
    PATH = "./saved_model/20210730153513/DenseNext32.ckpt"
    model = Classification(num_classes)
    model.load_state_dict(torch.load(PATH))
    torch.save(model.state_dict(), "./resave_" + model_name + ".pth")


if __name__ == "__main__":
    main()
