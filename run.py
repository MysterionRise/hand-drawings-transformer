from transformer.image import colour_detection


def main():
    colours = colour_detection(
        "resources/images/2022-04-27 19.30.35.jpg", number_of_colours=10
    )
    print(colours)


if __name__ == "__main__":
    main()
