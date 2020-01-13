from Dataset import DatasetSegment, Dataset

FOLDER_PATH = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\'
DATA_RECORD_NAME = '250NoBackground'
PATH_TO_ANNOTATED_DATA = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'

BACKGROUND = True
imageFolders = []
imageFolders.append('D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\')
# imageFolders.append('D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\subtractedBackgrounds\\')
imageFolders.append('D:\\BigData\\cellinfluid\\deformabilityObrazky\\')
# imageFolders.append('D:\\BigData\\cellinfluid\\deformabilityObrazky\\subtractedBackgrounds\\')
xmlFiles = []
xmlFiles.append(PATH_TO_ANNOTATED_DATA + 'tracks_1_300.xml')
xmlFiles.append(PATH_TO_ANNOTATED_DATA + 'deformabilityAnnotations.xml')

dataset = Dataset()
boundaries = []
boundaries.append(DatasetSegment(250, 300, 200, 250))
boundaries.append(DatasetSegment(50, 100, 30, 50))
dataset.generateTfRecord(FOLDER_PATH, '250And50V2', imageFolders, boundaries, xmlFiles)
boundaries.clear()
boundaries.append(DatasetSegment(0, 50, 200, 250))
boundaries.append(DatasetSegment(0, 50, 80, 100))
dataset.generateTfRecord(FOLDER_PATH, '250And501nd50', imageFolders, boundaries, xmlFiles)
boundaries.clear()
boundaries.append(DatasetSegment(50, 100, 100, 150))
boundaries.append(DatasetSegment(0, 50, 50, 70))
dataset.generateTfRecord(FOLDER_PATH, '250And502nd50', imageFolders, boundaries, xmlFiles)
boundaries.clear()
boundaries.append(DatasetSegment(100, 150, 200, 250))
boundaries.append(DatasetSegment(50, 100, 0, 20))
dataset.generateTfRecord(FOLDER_PATH, '250And503nd50', imageFolders, boundaries, xmlFiles)
boundaries.clear()
boundaries.append(DatasetSegment(150, 200, 0, 50))
boundaries.append(DatasetSegment(50, 100, 20, 40))
dataset.generateTfRecord(FOLDER_PATH, '250And504nd50', imageFolders, boundaries, xmlFiles)
