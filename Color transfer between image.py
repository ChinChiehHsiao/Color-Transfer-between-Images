import cv2
import numpy as np
import matplotlib.pyplot as plt


source = cv2.imread('s6.bmp')
reference = cv2.imread('t6.bmp')

source_LAB = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
reference_LAB = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

source_LAB_mean, source_LAB_std = cv2.meanStdDev(source_LAB)
reference_LAB_mean, reference_LAB_std = cv2.meanStdDev(reference_LAB)
source_LAB_mean, source_LAB_std = np.squeeze(source_LAB_mean), np.squeeze(source_LAB_std)
reference_LAB_mean, reference_LAB_std = np.squeeze(reference_LAB_mean), np.squeeze(reference_LAB_std)

result_LAB = (reference_LAB_std / source_LAB_std) * (source_LAB - source_LAB_mean) + reference_LAB_mean

result_BGR = cv2.cvtColor(np.clip(result_LAB, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

cv2.imwrite('result_image.bmp', result_BGR)


plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
plt.title('source')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
plt.title('reference')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_BGR, cv2.COLOR_BGR2RGB))
plt.title('result')
plt.axis('off')

plt.savefig('plt.png', bbox_inches='tight')
plt.show()
