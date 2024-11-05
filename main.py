from watershed import my_watershed
import matplotlib.pyplot as plt
import sys
import os


def show_watershed_results(img_path, show=False):
    contours, segments, df, n = my_watershed(img_path)

    fig, (ctrs, sgmts) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(f"{img_path}\nSegmemtation using the watershed algorithm\n Found {n} rocks")

    ctrs.imshow(contours)
    ctrs.axis('off')

    sgmts.imshow(segments)
    sgmts.axis('off')

    fig.tight_layout()

    df.to_csv("Dataframes/Echantillion1Mod2_301.csv", index=True, header=True)

    if show:
        plt.show()
        print(df)

    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("invalid number of arguments.\nUsage: python main.py <folder_path>\nfolder_path is the path to the folder containing the images")
    else :
        folder_path = sys.argv[1]
        if not os.path.isdir(folder_path):
            raise ValueError("The path provided is not a folder")
        for img_path in os.listdir(folder_path):
            df_means = show_watershed_results(os.path.join(folder_path, img_path))
        plt.show()
