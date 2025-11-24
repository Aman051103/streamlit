import streamlit as st
import pickle
import pandas as pd
import os
import zipfile

def get_info(book_ids):
    images = []
    infos = []
    previews = []
    titles = []
    for i in book_ids:
        book = books[books['Id']==i].image[books[books['Id']==i].index[0]]
        info = books[books['Id']==i].infoLink[books[books['Id']==i].index[0]]
        preview = books[books['Id']==i].previewLink[books[books['Id']==i].index[0]]
        title = books[books['Id']==i].Title[books[books['Id']==i].index[0]]
        images.append(book)
        infos.append(info)
        previews.append(preview)
        titles.append(title)
    return images, infos, previews, titles
    

def recommend(book):
    book_index = books[books['Title']==book].index[0]
    distances = similarity[book_index]
    recommended_books = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[:10]

    ids = []
    for i in recommended_books:
        ids.append(books.iloc[i[0]].Id)
    return ids

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(current_directory, 'booksDict.pkl')
file_path2 = os.path.join(current_directory, 'similarities.zip')

with zipfile.ZipFile(file_path2, 'r') as zip_ref:
    zip_ref.extractall('./')

# Path to the extracted pickled file
extracted_file_path = os.path.join('./', 'similarities.pkl')

books_dict = pickle.load(open(file_path1, 'rb'))
books = pd.DataFrame(books_dict)
similarity = pickle.load(open(extracted_file_path, 'rb'))

st.markdown("<h2 style='color: red;'>Book Recommender System</h2>", unsafe_allow_html=True)
st.write('Find top 10 books relevant to your requirements!')

label = "What would you like to read?"

selected_option = st.selectbox(label, ['']+list(books['Title'].values))
if st.button('Find'):
    images, infos, previews, titles = get_info(recommend(selected_option))

    html_elements = []
    for img, info, pre, title in zip(images, infos, previews, titles):
        html = f'<div style="display: inline-block; width: 40%; margin: 5% 5%"><a href="{pre}"><img src="{img}" width="100%" height="300px"></a><br>{title} <a href="{info}" style="text-decoration: none;"> - Info</a></div>'
        html_elements.append(html)

# Concatenate all HTML elements to display them in a single row
    html_row = ''.join(html_elements)

    # Display the HTML row in Streamlit
    st.markdown(html_row, unsafe_allow_html=True)
