import sys
import os
import pandas as pd
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QTableView, QHeaderView
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from watershed import my_watershed


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None


class ImageWidget(QWidget):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        self.initUI()

    def initUI(self):
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.fig, (self.ctrs, self.sgmts) = plt.subplots(1, 2, figsize=(9, 5))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.table = QTableView()
        self.layout.addWidget(self.table)

        self.update_image(self.img_path)

    def update_image(self, img_path):
        self.img_path = img_path

        contours, segments, df, n = my_watershed(self.img_path, )

        self.ctrs.imshow(contours)
        self.ctrs.axis('off')

        self.sgmts.imshow(segments)
        self.sgmts.axis('off')

        img_name = os.path.basename(self.img_path)
        self.fig.suptitle(f"{img_name}\nSegmentation using the watershed algorithm\n Found {n} rocks")
        self.canvas.draw()

        model = PandasModel(df)
        self.table.setModel(model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)


class MainWindow(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.current_index = 0

        self.initUI()

    def initUI(self):
        # size the window
        self.setGeometry(50, 50, 1500, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        self.button_layout.addWidget(self.next_button)

        self.image_widget = ImageWidget(self.image_files[self.current_index])
        self.layout.addWidget(self.image_widget)

        self.setWindowTitle("Watershed Segmentation Viewer")
        self.show()

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.image_widget.update_image(self.image_files[self.current_index])

    def show_next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.image_widget.update_image(self.image_files[self.current_index])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("invalid number of arguments.\nUsage: python segment_rocks_Qt.py <folder_path>\nfolder_path is the path to the folder containing the images")

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        raise ValueError("The path provided is not a folder")

    app = QApplication(sys.argv)
    main_window = MainWindow(folder_path)
    sys.exit(app.exec())
