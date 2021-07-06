from color_descriptor import ColorDescriptor
from config import get_postgres_connection
import glob
import cv2

# path to the directory that contains the images to be indexed
path_indexed = './folder/'

cd = ColorDescriptor((8, 12, 3))

list_ext_accept = ['png', 'jpg', 'jpeg']
for ext in list_ext_accept:
    connection = get_postgres_connection()
    cursor = connection.cursor()
    # scan all file images in folder and convert to colordescriptor vector and import to database
    for imagePath in glob.glob(args["dataset"] + "/**/*." + ext, recursive=True):
        try:
            image = cv2.imread(imagePath)
            features = cd.describe(image)
            features = [str(f) for f in features]
            features = ",".join(features)

            cursor.execute(f"""
                INSERT INTO image_descriptor (path, color_descriptor)
                VALUES ($${imagePath}$$, $${features}$$)
                ON CONFLICT DO NOTHING;
            """)
            connection.commit()
        except:
            print('Error')
