#CARTOON FILTER (Mia)
#get images 
images_cartoon = data_b3[b'data']
#reshape images 
images_cartoon = images_cartoon.reshape(len(images_cartoon),3,32,32).transpose(0,2,3,1) 

# Transform the image
  #blur images and display 
for image_cartoon in images_cartoon:
  #Converting to RGB
  image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_BGR2RGB)
  gray = cv2.cvtColor(image_cartoon, cv2.COLOR_BGR2GRAY)
  gray = cv2.medianBlur(gray, 5)
  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
  color = cv2.bilateralFilter(image_cartoon, 9, 250, 250)
  image_cartoon = cv2.bitwise_and(color, color, mask=edges)


# dispaly random images
# define row and column of figure
rows, columns = 5, 5
# take random image idex id
imageId = np.random.randint(0, len(images_cartoon), rows * columns)
# take images for above random image ids
images = images_cartoon[imageId]
# take labels for these images only
label_names = meta_data[b'labels']
labels = data_b3[b'labels']
labels = [labels[i] for i in imageId]

# define figure
fig=plt.figure(figsize=(10, 10))
# visualize these random images
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[i-1])
    plt.xticks([])
    plt.yticks([])
    plt.title("{}"
          .format(label_names[labels[i-1]]))
plt.show()
  
