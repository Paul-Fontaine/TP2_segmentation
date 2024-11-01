from watershed import my_watershed
import matplotlib.pyplot as plt


def watershed(img_path):
    contours, segments, df, n = my_watershed(img_path)

    fig, (ctrs, sgmts) = plt.subplots(1, 2)
    fig.set_size_inches(10, 6)
    fig.suptitle(f"Segmemtation using the watershed algorithm\n Found {n} rocks")

    ctrs.imshow(contours)
    ctrs.axis('off')

    sgmts.imshow(segments)
    sgmts.axis('off')

    fig.tight_layout()
    plt.show()

    df.to_csv("Dataframes/Echantillion1Mod2_301.csv", index=True, header=True)
    print(df)


if __name__ == "__main__":
    img_path = "Images/Echantillion1Mod2_301.png"
    watershed(img_path)

