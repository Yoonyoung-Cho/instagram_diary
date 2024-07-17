import streamlit as st
import glob

def load_images():
    image_files = glob.glob("instagram_diary/media/posts/*/*.jpg")
    manuscripts = []
    for image_file in image_files:
        image_file = image_file.replace("\\", "/")
        parts = image_file.split("/")
        if parts[1] not in manuscripts:
            manuscripts.append(parts[1])
    manuscripts.sort()

    return image_files, manuscripts

st.title("Demo Image Grid Display")
image_files, manuscripts = load_images()
view_manuscripts = st.multiselect("Select Manuscript(s)", manuscripts)
n = st.number_input("Select Grid Width", 1, 5, 3)

view_images = []
for image_file in image_files:
    if any(manuscript in image_file for manuscript in view_manuscripts):
        view_images.append(image_file)
groups = []
for i in range(0, len(view_images), n):
    groups.append(view_images[i:i+n])

for group in groups:
    cols = st.columns(n)
    for i, image_file in enumerate(group):
        cols[i].image(image_file)









# def load_images():
#     image_files = glob.glob("instagram_diary/media/posts/*/*.jpg")
#     st.write(len(image_files))
#     for image_file in image_files:
#         st.write(image_file)
#     return image_files#, manuscripts

# st.title('demo')
# image_files =load_images()
