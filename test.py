from DataManager import FileManager as fm

files = fm.enumerate_data_paths()

# print(len(list(files)))

for i, file in zip(range(1000), files):
    print(file)
    pic, vertices = fm.get_datum(file)
    print(vertices)

