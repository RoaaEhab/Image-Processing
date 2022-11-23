# Importing Image and ImageFilter module from PIL package
import numpy
from PIL import Image, ImageFilter
	
# # creating a image object
# im1 = Image.open(r"Images\textbw.png")
	
# # applying the unsharpmask method
# im2 = im1.filter(ImageFilter.UnsharpMask(radius = 3, percent = 200, threshold = 5))
	
# im2.show()



def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final
for i in range(original_image_height):
            for j in range(original_image_width):
                for k in range(kernel_size):
                    for l in range(kernel_size):
                       temp.append(padded_image_array[k + i,l + j])
                temp.sort()
                filtered_image_array[i + kernel_size//2,j + kernel_size//2] = temp[len(temp) // 2]
        filtered_image= Image.fromarray(np.uint8(filtered_image_array))
def median(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                # if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                #     for c in range(filter_size):
                #         temp.append(0)
                # else:
                #     if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                #         temp.append(0)
                #     else:
                        for k in range(filter_size):
                            temp.append(data[i +k][j + k])

            temp.sort()
            data_final[i + filter_size//2,j + filter_size//2] = temp[len(temp) // 2]
            temp = []
    return data_final


def main():
    x = numpy.zeros((3,3))
    x[0,0] = 5
    x[0,1] = 6
    x[0,2] = 2
    x[1,0] = 9
    x[1,1] = 1
    x[1,2] = 3
    x[2,0] = 6
    x[2,1] = 4
    x[2,2] = 4
    print(x)
    padded_image_width=3 + 3 -1
    padded_image_height=3 + 3 -1
    # padded image
    padded_image_array = numpy.pad(x, 3//2)
    # img = Image.open(r"Images\noisyimg.png").convert(
    #     "L")
    # arr = numpy.array(img)
    removed_noise = median_filter(x, 3) 
    print(removed_noise)
    # img = Image.fromarray(removed_noise)
    # img.show()


main()
