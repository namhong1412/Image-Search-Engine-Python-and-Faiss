> Trying to search for images based on text and tags sucks and using text and keywords to describe something inherently visual is a real pain.

Then, I stopped my manual search and opened up a code editor.

### What's an Image Search Engine?

I mean, you're not using text or your query, you're instead using an image.
Sounds pretty hard, and how to quantify the contents of an image to make it searchable.

### Some important terms

When building an Image Search Engine, I'll first have an **indexed dataset**. 
To get an **indexed dataset**, I need to process quantify our dataset by utilizing an image descriptor to extract **features** from each image. And an ** image descriptor** defines the algorithm that I'm utilizing to describe our image.

For example:

- The mean and standard deviation of each **Red, Green, and Blue** channel, respectively,
- The statistical moments of the image to characterize shape.
- The gradient magnitude and orientation describe both shape and texture.

**Features** are the output of an image descriptor. So when you put an image into an **image descriptor**, you'll get **features** out the other end.

**Features** (or **Feature Vectors**) are just a list of numbers used to abstractly represent and quantify images.

Take a look at the example figure below:
![bovw_multiple_feature_vectors.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626060340049/fGYEUM4Fs.png)
> The pipeline of an image descriptor. An input image is presented to the descriptor, the image descriptor is applied, and a feature vector (i.e a list of numbers) is returned, used to quantify the contents of the image.

Feature Vector can then be compared for similarity by using the distance metric or similarity function. Distance metric and similarity function take two features vectors as inputs then output a number that represents how "similar" the two feature vectors are.

![comparing_images.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1626063212546/W6a7UdUxV.jpeg)
> Given two feature vectors, a distance function is used to determine how similar the two feature vectors are. The output of the distance function is a single floating-point value used to represent the similarity between the two images.

### The 4 Steps of Any CBIR System

1. **Define image descriptor**: You need to decide what aspect of the image you want to describe. Are you interested in the color, the shape... of the image? Or do you want to characterize texture?

2. **Index dataset**: you are to apply this image descriptor to each image in your dataset, extract features from these images, and write the features to CSV file, RDBMS, Redis, etc. so that they can be later compared for similarity.
![preprocessing_and_indexing.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1626064095796/f6j-nSjqZ.jpeg)

3. **Define similarity metric**: How are you going to compare them? Popular choices include the **Euclidean distance, Cosine distance, and chi-squared distance**, but the actual choice is highly dependent on **your dataset** and the types of features you extracted.

4. **Search**: The user will submit a query image to your system (**Form upload or API**) and you need be to extract features from this query image and then apply your similarity function to compare the query features to the features already indexed finally return the most relevant results according to your similarity function.
![searching.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1626064111199/iWOkk5O3S.jpeg)

### Our dataset of my company
Here are a few samples from the dataset:
![Screen Shot 2021-07-12 at 11.33.51.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626064442386/KnLSa6PmTX.png)

### Step 1: Define Image Descriptor
Instead of using a standard color histogram, I'll be using a 3D histogram in the HSV color space (Hue, Saturation, Value). 
![1024px-RGB_color_solid_cube.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626072505267/56kwdvnTK.png)

The reason I don't use RBG color (standard color) is RGB values are simple to understand, the RGB color space fails to mimic how humans perceive color.
![HSV_color_solid_cylinder.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626072378339/QAdh-iNqd.png)

Or you can use such as the CIE L*a*b* and CIE XYZ color spaces.

Histograms are used to give a rough sense of the density of pixel intensities in an image. So, now I need to define the number of Bins for histograms.

The histogram will estimate the probability density of the underlying function, or in this case, the probability P of a pixel color C occurring in our image.

Here, in this post, I'll be utilizing a 3D color histogram in the HSV color space with 8 bins for the Hue channel, 12 bins for the saturation channel, and 3 bins for the value channel, yielding a total feature vector of dimension 8 x 12 x 3 = 288. This means all images will be abstractly represented and quantified using only a list of 288 floating-point numbers.

Now, I creating a new file, name it `colordescriptor.py`
```
import numpy as np
import cv2
import imutils

class color_descriptor:
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):
		# convert the image to the HSV color space
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
``` 
So now the hard work starts.
Instead of computing a 3D HSV color histogram for the **entire image**, let's instead compute a 3d HSV color histogram for **different regions **of the image. Because using region-based histograms rather than global histograms allows us to simulate locality in color distribution.

For my image descriptor, I'm going to divide my image into five different regions:

1. The top-left corner
2. The top-right corner
3. The bottom-right corner
4. The bottom-left corner
5. The center of the image

After is all code of `color_descriptor class` of file `colordescriptor.py`

```
import numpy as np
import cv2
import imutils


class color_descriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # top-left, top-right, bottom-right, bottom-left corner
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # elliptical mask in the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image
            hist = self.histogram(image, cornerMask)

            # then update the feature vector
            features.extend(hist)

        # extract a color histogram from the elliptical region
        hist = self.histogram(image, ellipMask)

        # then update the feature vector
        features.extend(hist)

        # return the feature vector
        return features
```

The histogram method then returns a color histogram representing the current region, which we append to our features list.

```
def histogram(self, image, mask):
    # extract a 3D color histogram from the masked region
	hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()

	# return the histogram
	return hist
```

## Step 2: Create Database and Extracting Features

Create a new database, here I using PostgreSQL and create database name `data_lake`.

Now, I created table name `image_descriptor`:

```
create table image_descriptor
(
    id               bigserial     not null unique,
    path             varchar(1000) not null unique,
    color_descriptor text          not null,
    status           int           not null default 1,
    created_at       timestamp     not null default current_timestamp,
    updated_at       timestamp     not null default current_timestamp,
    primary key (id)
);
```
Then, create a new file `config.py` use to connect to the database:
```
import psycopg2

HOSTNAME = 'localhost'
USERNAME = 'postgres'
PASSWORD = 'your_password'
DATABASE_NAME = 'data_lake'
PORT = 5432


def get_postgres_connection():
    try:
        connection = psycopg2.connect(
            user=USERNAME,
            password=PASSWORD,
            host=HOSTNAME,
            port=PORT,
            database=DATABASE_NAME)
        return connection
    except (Exception, psycopg2.Error) as error:
        message = f"get_postgres_connection {error}"
        return abort(400, message)
```

Open up a new file, name it `index.py`:

```
from colordescriptor import color_descriptor
from config import get_postgres_connection
import argparse
import glob
import cv2

path_dataset = "/home/admin/dataset"

connection = get_postgres_connection()
cursor = connection.cursor()

cd = color_descriptor((8, 12, 3))

# accept extension file of image
ext = ['png', 'jpg', 'jpeg']

for e in ext:
    for imagePath in glob.glob(path_dataset + "/**/*." + e, recursive=True):
        try:
            image = cv2.imread(imagePath)

            features = cd.describe(image)

            features = [str(f) for f in features]
            features = ",".join(features)
            
            # insert features to database
            cursor.execute(f"""
                    INSERT INTO image_descriptor (path, color_descriptor)
                    VALUES ($${imagePath}$$, $${features}$$)
                    ON CONFLICT DO NOTHING;
                """)
            connection.commit()
        except:
            print(imagePath)
```

Now I open up a shell and issue the following command:
`python3 index.py`

In the processing, you can visit your database to see the result image descriptor.

![Screen Shot 2021-07-12 at 15.03.54.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626077043946/wFI_GclqV.png)

### Step 3: The Searcher

First, I'll talk to you about Faiss.
Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/NumPy. Some of the most useful algorithms are implemented on the GPU. It is developed by Facebook AI Research.

Here, I use Faiss to know the similarity between features, that is, now we perform the calculation to subtract two feature vectors to get the distance between them. The smaller the distance, the higher the odds of them being the same.

Create a new file, it name `search.py`
```
from colordescriptor import color_descriptor
from config import get_postgres_connection
import numpy
import faiss
import ast
import argparse
import cv2
import datetime
import forr


def get_total_dataset_dtb():
    connection = get_postgres_connection()
    cursor = connection.cursor()

    cursor.execute(f"""
        select count(*) from image_descriptor;
    """)
    connection.commit()
    rows = cursor.fetchall()
    return int(rows[0][0])


def get_dataset_vector(page, page_size=20000, offset=None):
    connection = get_postgres_connection()
    cursor = connection.cursor()

    if offset is None:
        limit = f" LIMIT {page_size} offset {(page - 1) * page_size}"
    else:
        limit = f" LIMIT {page_size} offset {offset + ((page - 1) * page_size)}"

    cursor.execute(f"""
        select color_descriptor from image_descriptor order by id {limit};
    """)
    connection.commit()
    rows = cursor.fetchall()
    list_vector = []
    for row in rows:
        features = [float(x) for x in row[2].split(',')]
        list_vector.append(features)

    return numpy.array(list_vector)


def get_feature_vector(path_image):
    # define color descriptor
    cd = color_descriptor((8, 12, 3))

    # create ndarray by vectors of image file
    query = cv2.imread(path_image)
    features = cd.describe(query)
    features_vector = [features]

    return numpy.array(features_vector).astype('float32')


def get_index_vector():
    dimension = 1440
    index = faiss.IndexFlatL2(dimension)
    index = faiss.read_index("train.index")
    print(index.ntotal)
    total_dataset_dtb = get_total_dataset_dtb()
    vector_total = index.ntotal
    if total_dataset_dtb > vector_total:
        for page in range(1, 100):
            page_size = 10000

            # find dimention vectors
            if vector_total > 0:
                data_vector = get_dataset_vector(page, page_size, vector_total)
            else:
                data_vector = get_dataset_vector(page, page_size)

            if len(data_vector) > 0:
                vector_total = vector_total + page_size
                # build the index by dimension and add vectors to the index
                data_vector = data_vector.reshape(data_vector.shape[0], -1).astype('float32')
                index.add(data_vector)
                # print(index.ntotal)
            else:
                break

        # write index file to disk
        faiss.write_index(index, 'train.index')
    print(index.ntotal)
    return index


def search_vector(path_image, total_results):
    index = get_index_vector()
    features_vector_search = get_feature_vector(path_image)

    # search image by the feature vector
    D, I = index.search(features_vector_search, total_results)  # actual search
    index.reset()
    return I


def get_image_similar(path_image):
    print(datetime.datetime.now())
    connection = get_postgres_connection()
    cursor = connection.cursor()

    total_results = 10
    result_vector = search_vector(path_image, total_results)
    query = ""
    for j in range(total_results):
        query += f"(select id, path from image_descriptor order by id LIMIT 1 offset {result_vector[0][j]}) union all "

    cursor.execute(query[:-11])
    connection.commit()
    rows = cursor.fetchall()
    print(datetime.datetime.now())
    return rows

if __name__ == '__main__':
    path_image = 'your_path'
    get_image_similar(path_image)
```

You can see, after each time, I always save the feature vectors I got from the database to the file **train.index** to use the next time.
By cleverly saving vectors, you can search up to** 1 second**.

### Test My CBIR System
Replace your path image file in file search.py:
```
if __name__ == '__main__':
    path_image = 'your_path'
    get_image_similar(path_image)
```
Open up your terminal, navigate to the directory where your code lives, and issue the following command: `python3 search.py`

And here is my result after trying to perfect a complete interface:

**Example 1**

Input:
![custompillow-family-010920-witch-fr-black.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626079319065/SPmSRBml6.png)

Output:
![Screen Shot 2021-07-12 at 15.42.53.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626079391518/3SvFTdBDW.png)
The returned results are ranked and we can see similar images appear here.

**Example 2**

Input:
![Screen Shot 2021-07-12 at 15.51.58.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626079933604/SFngWf_Ew.png)

Output:
![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1626080025525/RHrflCsLl.png)
These search results are also quite good.
Thanks for reading. good luck 🌝🌝🌝
