import numpy
import faiss
from config import get_postgres_connection
from color_descriptor import ColorDescriptor


def dataset_vector():
    connection = get_postgres_connection()
    cursor = connection.cursor()
    cursor.execute(f"""
        select * from image_descriptor;
    """)
    connection.commit()
    rows = cursor.fetchall()
    list_vector = []
    for row in rows:
        features = [float(x) for x in row[2].split(',')]
        list_vector.append(features)
    result = numpy.array(list_vector)
    return result


# define color descriptor
cd = ColorDescriptor((8, 12, 3))

# path file image need search
path_iamge_search = 'custompillow-family-010920-witch-fr-black.png'

# create ndarray by vectors of image file
query = cv2.imread(path_iamge_search)
features = cd.describe(query)
features_vector = [features]
features_vector_search = numpy.array(features_vector).astype('float32')

# find dimention vectors
data_vector = dataset_vector()
data_vector = data_vector.reshape(data_vector.shape[0], -1).astype('float32')
d = data_vector.shape[1]  # dimension

# build the index by dimension and add vectors to the index
index = faiss.IndexFlatL2(d)
index.add(data_vector)
print(index.ntotal)

# see 4 nearest neighbors
k = 4
D, I = index.search(features_vector_search, k)  # actual search
print(I[:5])  # neighbors of the 5 first queries
print(I[-5:])  # neighbors of the 5 last queries
