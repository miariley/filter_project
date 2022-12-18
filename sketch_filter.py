#Sketch Filter with User Input 
#filter image 
def sketchFilter(image):
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
  #sharpen 
  kernel_size = 11
  kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  img_sharpen = cv2.filter2D(img_gray, -1, kernel)
  #canny edge 
  sketch = cv2.Canny(img_sharpen,10,20)
  plt.figure(figsize=(18, 8))
  plt.subplot(121)
  plt.imshow(image)
  plt.title('input')
  plt.axis('off')
  plt.subplot(122)
  plt.imshow(sketch)
  plt.title('sketched image')
  plt.axis('off')

sketchFilter(image)
