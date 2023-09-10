
# Comics-AudioBooks

For the longest time, literature books have been printed and distributed physically; however with the advancement of file distribution platforms, literature books were being digitally published and distributed as PDF files. With the constant improvements and developments of technology, literature books have advanced to a point where the written words are read aloud as an audiobook. Hence, with the popularity of comic books, which is another form of written literature, that utilizes both words and images to convert and present information to the reader as another form of entertainment. Thus, similar to the advancement of literature books transformed into audiobooks, the development of comic books transformed into audiobooks can expand the accessibility to a larger audience.

## Demo

- Voice is not present in this gif

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/output-comic.gif?raw=true)

- Random comic book page pic taken from phone and uploaded in the app

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/lib.gif?raw=true)
## Authors

- [@Bailey Wang]()
- [@Lakshmi Naga Meghana Polisetty]()
- [@Sakshi Tongia](https://github.com/sakshitongia)
- [@Vaibhav Yalla](https://github.com/yallavaibhav)



## Acknowledgements

We would like to extend our largest gratitude to our advisor and mentor Dr. Eduardo Chan for providing his knowledge, advice, and expertise in order to help us complete this project. His continuous instruction and support ensured that the project would be successfully completed. An extended acknowledgment to our family members and spouses for their overwhelming support and motivation throughout this project.
## Motivation
- Evolution of Reading experience since the 80s. 

- New technology devices like Kindles have made reading easier and simpler.

- Audiobook industry has seen rapid growth in recent years. 

- New emerging smartphone technology platforms like Audible, Scribd, and Kobo have made the users feel “Listening is the New Reading”

- Anime creators have forged plenty of new styles, genres, and technology. 

- Growing trend for graphic novels to make their way from printed formats to digital and streaming platforms utilizing Deep Learning and Artificial Intelligence (AI).

- Improve AI- Generation experience of users transitioning from e-book readers to Audiobook listeners

## Methodology

#### Algorithms and services used
- Yolov5
- Google Vision API
- SV2TTS
- Python
- Pytorch
- SQL
- BigQuery

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20173957.png?raw=true)


![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174052.png?raw=true)
## Panel Detection


![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174212.png?raw=true)


### Yolov5
- Better visualization for the users
- Yolov5 with Dark net format.
- Trained with 25 epochs.
- Number of classes = 1
- Customized Yolov5 model, by extracting only the detected parts of the image and sorting them in a systematic comic reader order.
- These cropped parts are further used for characters and speech detection.



![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174226.png?raw=true)

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174242.png?raw=true)
## Character Detection

### Yolo V5

- Model was trained on 120 Epochs
- Image size - 640*640
- Classes : 30+ characters
- Number of panel images : 4600
- Optimizer - SGD, Learning Rate - 0.005, Weight Decay 0.0005
- Implemented same sorting logic for character location.

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174304.png?raw=true)

### Faster RCNN

- Pretrained model on VGG16 on custom dataset of characters
- Used embedding pre-trained on Faster RCNN to train it on panel images
- Classes : 30+ characters
- Model was trained on 100 epochs 
- Optimizer - SGD, Learning Rate - 0.005, Weight Decay 0.0005

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174319.png?raw=true)

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174334.png?raw=true)
## Speech Balloon Detection

### Efficient Net
- Model was trained on 2000 Epochs
- Train - 314; Test - 51; Validation - 87 Images
- Hyperparameters
- Optimizer - Adam
- Weight Decay - 0.0005
- Learning Rate - 0.001
- Classes - 6

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174406.png?raw=true)

### Yolo V5
- Model was trained on 220 Epochs
- Train - 314; Test - 51; Validation - 87 Images
- Hyperparameters
- Optimizer - SGD
- Weight Decay - 0.0005
- Learning Rate - 0.001
- Classes - 6
- Classes - General_speech, Roar, blast_sound, hit_sound, narration_speech, thought_speech

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174351.png?raw=true)


![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174420.png?raw=true)
## Text Recognization

- Text Recognition Models developed using EasyOCR and KerasOCR
- Both models required to train the models for different types of fonts
- After Font training, 100 Epochs were run in order to train the model
- Both models require ~60 minutes to train

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174442.png?raw=true)

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174455.png?raw=true)
## Speech Cloning

- Each character in the comic book has his/her own authentic voice.
Data from cartoons, movies are collected and edited using Audacity tool.
- Encoder: Input the data, converts into Mel Spectrogram. Encoder focuses on  accent, tone, pitch etc. Creates a embedding using these features.
- Synthesizer: Generates mel spectrograms for new data
Vocoder: Converts spectrograms to audio data.


![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174548.png?raw=true)

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174602.png?raw=true)
## Data Collection
## UI/ Frontend
- HTML, CSS, Bootstrap, Flask

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174628.png?raw=true)

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174647.png?raw=true)


![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/Screenshot%202023-09-09%20174718.png?raw=true)
## Conclusion

- Developed Deep Learning Algorithms for each of the Panel, Character, Speech Bubbles, Text Recognition Problems.
- Providing a scan of a physical comic book and uploading will also give results.
- A non-colored comic book will provide results in the web-application.
- Moviepy is used for merging the output and images.
- Deployed application using Flask/Cloud (GCP).
- Produced a Comic Audiobook Application.
## Societal Impact  
- Allow Publishers & Creators to expand audience, create better business opportunities.
- Can be used as a learning tool for visual impairments/learning disabilities.
## Future Works
- Model improvements (Detection, Accuracy, Speed)
- Develop into a mobile application (for on the go accessibility)
- Expand scope of project (more genres, different languages)
- Consider Text bubble tails instead of only center point
- Provide more character voices to be trained
- Integrate deepfake with our application for a better experience 

![App Screenshot](https://github.com/yallavaibhav/Comic-AudioBooks/blob/master/Screenshots/output-comic1.gif?raw=true)
## References


- Alvin, T. P. (2022, November 29). Introduction to Text Summarization with ROUGE Scores. Medium. https://towardsdatascience.com/introduction-to-text-summarization-with-rouge-scores-84140c64b471
- Andrews, D., Baber, Efremov, S., & Komarov, M. (2012, May). Creating and using interactive narratives: reading and writing branching comics. ACM Digital Library. Retrieved February 23, 2023, from https://dl.acm.org/doi/abs/10.1145/2207676.2208298?casa_token=8U38xJKygkYAAAAA:YVG9dm2v4EWAYdmcGsHDSXx7QaKKJu_rqvIOxe53cCdrs8VP4FUs3i3xB71pQAy19S1GiChk54M8
- Arai, K., & Tolle, H. (2011, February). Method for Real Time Text Extraction of Digital Manga Comic. Cscjournals. Retrieved October 14, 2022, from https://www.cscjournals.org/manuscript/Journals/IJIP/Volume4/Issue6/IJIP-290.pdf
- Augereau, O. (2018, June 26). A Survey of Comics Research in Computer Science. MDPI. Retrieved February 23, 2023, from https://www.mdpi.com/2313-433X/4/7/87
- Bivona, A. (2021, December 16). A Tutorial on Scraping Images from the Web Using BeautifulSoup. Medium. Retrieved October 5, 2022, from https://towardsdatascience.com/a-tutorial-on-scraping-images-from-the-web-using-beautifulsoup-206a7633e948
- Burie, J. (2015, August 30). Robust Frame and Text Extraction from Comic Books. https://www.academia.edu/15282107/Robust_Frame_and_Text_Extraction_from_Comic_Books?email_work_card=view-paper 
- Chen, C., Li, Z., Chen, H., & Li, G. (2021). Fast and Accurate Speech Balloon Detection in Comics Using EfficientNet. IEEE Transactions on Multimedia, 23, 1435-1448.
- Chu, W., & Li, W. (2019, February). Manga face detection based on deep neural networks fusing global and local information. ScienceDirect. Retrieved October 7, 2022, from https://www.sciencedirect.com/science/article/pii/S0031320318303066
- Cloud Storage Pricing. (n.d.). Google Cloud. Retrieved October 5, 2022, from https://cloud.google.com/storage/pricing
- Coumar, N. (2021, December 15). Optical Character Recognition(OCR) — Image, Opencv, pytesseract and easyocr. Medium. Retrieved April 7, 2023, from https://medium.com/@nandacoumar/optical-character-recognition-ocr-image-opencv-pytesseract-and-easyocr-62603ca4357
- Dolphin, R. (2020, October 21). LSTM Networks | A Detailed Explanation. Towards Data Science. 
- Doshi, K. (2022, January 6). Foundations of NLP Explained — Bleu Score and WER Metrics. Medium. https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b
- Dubray, D., & Laubrock, J. (2019, September 1). Deep CNN-Based Speech Balloon Detection and Segmentation for Comic Books. IEEE Conference Publication | IEEE Xplore. Retrieved October 1, 2022, from https://ieeexplore.ieee.org/abstract/document/8977973
- Dutta, A., & Biswas, S. (2019, September). CNN Based Extraction of Panels/Characters from Bengali Comic Book Page Images. 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW). https://doi.org/10.1109/icdarw.2019.00012
- Getting Started with Image Preprocessing in Python. (n.d.). Engineering Education (EngEd) Program | Section. https://www.section.io/engineering-education/image-preprocessing-in-python/
- Ghorbel, A., Ogier, J., & Vincent, N. (2015). Text extraction from comic books. ResearchGate. https://doi.org/10.13140/RG.2.1.3631.9203
- Guerin, C., Rigaud, C., Mercier, A., Ammar-Boudjelal, F., Bertet, K., Bouju, A., Burie, K., Louis, G., Ogier, J., & Revel, A. (2013, August 1). eBDtheque: A Representative Database of Comics. IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/document/6628793
- Guide  |  TensorFlow Core. (2021, September 23). TensorFlow. https://www.tensorflow.org/guide
- Ieamsaard, J., Charoensook, S., & Yammen, S. (2021, March 10). Deep Learning-based Face Mask Detection Using YoloV5. IEEE Xplore. https://ieeexplore-ieee-org.libaccess.sjlibrary.org/document/9440346 
- ImageNet. (n.d.). https://www.image-net.org/index.php 
- JaidedAi. (2022, September 15). EasyOCR. Github. Retrieved April 7, 2023, from https://github.com/JaidedAI/EasyOCR
- Kaori, S. (2003). Yamato No Hane (1st ed., Vol. 1) [Digital]. Kodansha.
- Kasper-Eulaers, M., Hahn, N., Berger, S., Sebulonsen, T., Myrland, Ø., & Kummervold, P. E. (2021). Short Communication: Detecting Heavy Goods Vehicles in Rest Areas in Winter Conditions Using YOLOv5. Algorithms, 14(4), 114. https://doi.org/10.3390/a14040114
- Kaur, D., & Kaur, Y. (2015, May 5). Various Image Segmentation Techniques: A Review. International Journal of Computer Science and Mobile Computing, Vol. 3(Issue. 5), 809–814. https://ijcsmc.com/docs/papers/May2014/V3I5201499a84.pdf 
- Kishimoto, M. (2003, August 16). Naruto, Vol. 1: Uzumaki Naruto (1st Edition). VIZ Media LLC.
- Kiyoharu, A. (2020). Manga109 [Dataset]. The University of Tokyo. http://www.manga109.org/
- Klatt, D. H. (1987, September). Review of text to speech conversion for English. The Journal of the Acoustical Society of America, 82(3), 737–793. https://doi.org/10.1121/1.395275 
- Laubrock, J., & Dubray, D. (2019). Multi-class semantic segmentation of comics: A U-Net based approach. In Graphics Recognition (GREC) Workshop, International Conference on Document Analysis and Recognition.
- Laubrock, J., & Dunst, A. (2019, November 8). Computational Approaches to Comics Analysis. Wiley Online Library. Retrieved February 23, 2023, from https://onlinelibrary.wiley.com/doi/full/10.1111/tops.12476
- Laubrock, J., & Dunst, A. (2020). Computational Approaches to Comics Analysis. Topics in Cognitive Science, 12(1), 274–310. https://doi.org/10.1111/tops.12476
- Lee, M. J., Rhee, C., & Lee, C. (2022, January 18). HSVNet: Reconstructing HDR Image from a Single Exposure LDR Image with CNN. MDPI. Retrieved October 14, 2022, from https://www.mdpi.com/2076-3417/12/5/2370
- Lenadora, D., Ranathunge, R., Samarawickrama, C., De Silva, Y., Perera, I., & Welivita, A. (2020, March 5). Comic Digitization through the Extraction of Semantic Content and Style Analysis. IEEE Conference Publication | IEEE Xplore. Retrieved October 4, 2022, from https://ieeexplore.ieee.org/abstract/document/9023647?casa_token=ByagDsxKQxMAAAAA:GhaQwaxsLO-04kexc2GPrtLnA3LDSG4gLj_4Re9F-tYBq6N8TxTto8uyfLhQdWVB0DOIirN3vA
- Leung, K. (2022, January 6). Evaluate OCR Output Quality with Character Error Rate (CER) and Word Error Rate (WER). Medium. https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510 
- Listening is the new reading. (2022). Reply. Retrieved October 4, 2022, from https://www.reply.com/en/r20/listening-is-the-new-reading 
- Liu, X., & Tang, Z. (2016, January 3). An R-CNN Based Method to Localize Speech Balloons in Comics. SpringerLink. Retrieved October 2, 2022, from https://link.springer.com/chapter/10.1007/978-3-319-27671-7_37?error=cookies_not_supported&code=55acb095-e373-4e9e-8a79-9d224c010871
- Madhugiri, D. (2022, September 2). Extract Text from Images Quickly Using Keras-OCR Pipeline. Analytics Vidhya. Retrieved April 7, 2023, from https://www.analyticsvidhya.com/blog/2022/09/extract-text-from-images-quickly-using-keras-ocr-pipeline/#:~:text=What%20is%20Keras%20OCR%3F,%2C%20and%20image%2Donly%20pdf
- McCloud, S., Martin, M. (1994). Understanding comics : the invisible art. Germany: HarperCollins.
- McMillan, G. (2023, February 1). Graphic novel sales up a staggering 110.2% in bookstores since the pandemic. Popverse. https://www.thepopverse.com/graphic-novel-sales-up-a-staggering-1102-in-bookstores-since-the-pandemic
- Memon, J., Sami, M., Khan, R. A., & Uddin, M. (2020). Handwritten Optical Character Recognition (OCR): A Comprehensive Systematic Literature Review (SLR). IEEE Access, 8, 142642–142668. https://doi.org/10.1109/access.2020.3012542
- Meyer, D., Wiertlewski, Peshkin, M., & Colgate, J. (2014, March 20). Dynamics of ultrasonic and electrostatic friction modulation for rendering texture on haptic surfaces. Ieeexplore. Retrieved February 23, 2023, from https://ieeexplore.ieee.org/abstract/document/6775434?casa_token=5SY8U18Bp94AAAAA:KjZsH7Wp_2sdjl1a24jkNaaGt7jpcq6jih8RzTg2kMrs-tyZRFpe-OnAOQjMivgCU_0YLL3jyA
- Mittal, A. (2021, December 4). Understanding RNN and LSTM - Aditi Mittal. Medium. https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e
- MS COCO Dataset: Using it in Your Computer Vision Projects. (2022, October 25). Datagen. https://datagen.tech/guides/image-datasets/ms-coco-dataset-using-it-in-your-computer-vision-projects/ 
- MSDA Admin. (2020, January 16). What should be my computer configuration for taking classes? / Master of Science in Data Analytics. San Jose State University. Retrieved October 5, 2022, from https://blogs.sjsu.edu/msda/2020/01/16/what-should-be-my-computer-configuration-for-taking-classes/
- Nakashima, T. (2015, March 20). Manga Panel Layout Basics – “2×4” Grid – Japanese Manga 101 – #010. SILENT MANGA AUDITION®. https://www.manga-audition.com/japanesemanga101_010/
- Nguyen, N., Rigaud, C., & Burie, J. (2018, January 29). Comic Characters Detection Using Deep Learning. IEEE Xplore. Retrieved October 15, 2022, from https://ieeexplore-ieee-org.libaccess.sjlibrary.org/document/8270235
- Nguyen, N. V., Rigaud, C., & Burie, J. C. (2018, July 2). Digital Comics Image Indexing Based on Deep Learning. Journal of Imaging, 4(7), 89. https://doi.org/10.3390/jimaging4070089
- Nguyen Nhu, V., Rigaud, C., & Burie, J. C. (2019, September). What do We Expect from Comic Panel Extraction? 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW). https://doi.org/10.1109/icdarw.2019.00013
- Nonaka, S., Sawano, T., & Haneda, N. (2011, December 15). Development of “GT-Scan”, the Technology for Automatic Detection of Frames in Scanned Comic. Fujifilm. Retrieved October 14, 2022, from https://asset.fujifilm.com/www/jp/files/2019-12/55c580fb09e0e4d4d4a91b0fbb3ab051/ff_rd057_010_en.pdf
- Ogawa, T., Otsubo, A., Narita, R., Matsui, Y., Yamasaki, T., & Aizawa, K. (2018, March 26). Object Detection for Comics using Manga109 Annotations. Arxiv. Retrieved October 7, 2022, from https://arxiv.org/pdf/1803.08670v2.pdf
- OpenCV: Canny Edge Detection. (n.d.). https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- OpenCV: OpenCV modules. (2022). OpenCV. https://docs.opencv.org/4.x/
- Pathak, A. R., Pandey, M., & Rautaray, S. (2018). Application of Deep Learning for Object Detection. Procedia Computer Science, 132, 1706–1717. https://doi.org/10.1016/j.procs.2018.05.144 
- Piriyothinkul, B., Pasupa, K., & Sugimoto, M. (2019, January 23). Detecting Text in Manga Using Stroke Width Transform. IEEE Conference Publication | IEEE Xplore. Retrieved October 1, 2022, from https://ieeexplore.ieee.org/abstract/document/8687404?casa_token=YJqDAhkUMGcAAAAA:Qe9Gh9ev9WccYJqn6AQ9X816soLkuFjaPiF2AFhTrGc-UV0mjMLrqKfH6zd5xVFjIq-eHox7uA
- PNG vs. SVG: What are the differences? | Adobe. (2022). Adobe. https://www.adobe.com/creativecloud/file-types/image/comparison/png-vs-svg.html
- Ponsard, C., & Fries, V. (2008). An Accessible Viewer for Digital Comic Books. SpringerLink. https://link.springer.com/chapter/10.1007/978-3-540-70540-6_81?error=cookies_not_supported&code=9b76392b-9497-4d1e-8467-1beb757c8116
- PyTorch documentation — PyTorch 1.13 documentation. (2022). https://pytorch.org/docs/stable/index.html
- Qin, X., Zhou, Y., He, Z., Wang, Y., & Tang, Z. (2017, November 1). A Faster R-CNN Based Method for Comic Characters Face Detection. IEEE Conference Publication | IEEE Xplore. Retrieved October 7, 2022, from https://ieeexplore.ieee.org/abstract/document/8270109
- Rath, S. (2020, August 31). Evaluation Metrics for Object Detection. Debugger Cafe. https://debuggercafe.com/evaluation-metrics-for-object-detection/
- Redmon, J., & Farhadi, A. (2018, April 8). YOLOv3: An Incremental Improvement. Arxiv. https://arxiv.org/pdf/1804.02767.pdf 
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: towards real-time object detection with region proposal networks. Neural Information Processing Systems, 28, 91–99. https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf 
- Rigaud, C., Burie, J., Ogier, J., Karatzas, D., & Weijer, J. (2013, August 23). An Active Contour Model for Speech Balloon Detection in Comics. IEEE Conference Publication | IEEE Xplore. Retrieved October 2, 2022, from https://ieeexplore.ieee.org/abstract/document/6628812?casa_token=625k48uKj_0AAAAA:Rl54DRyoJZDy2kDknIs0pxPf7fK-zllEt3wwB1ZF1WUE9wN8P2A49Gbm1GrMPK3kExednzUQg-g
- Rigaud, C., Burie, J., & Ogier, J. (2017, January 8). Text-Independent Speech Balloon Segmentation for Comics and Manga. SpringerLink. https://link.springer.com/chapter/10.1007/978-3-319-52159-6_10#Sec5 
- Rigaud, C., Thanh, N., Burie, J., Ogier, J., Iwata, M., Imazu, E., & Kise, K. (2015, August 23). Speech balloon and speaker association for comics and manga understanding. IEEE Xplore. Retrieved April 7, 2023, from https://ieeexplore-ieee-org.libaccess.sjlibrary.org/document/7333782
- Rigaud, C., Nguyen, N., & Burie, J. (2021, February 25). Text Block Segmentation in Comic Speech Bubbles. SpringerLink. Retrieved October 1, 2022, from https://link.springer.com/chapter/10.1007/978-3-030-68780-9_22?error=cookies_not_supported&code=ab822542-8fcf-482c-8f04-dc1204be97e1
- Roggia, C., & Persia, F. (2020, December). Extraction of Frame Sequences in the Manga Context. 2020 IEEE International Symposium on Multimedia (ISM). https://doi.org/10.1109/ism.2020.00023
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Lecture Notes in Computer Science, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28
- Rosebrock, A. (2021, August 18). Getting started with EasyOCR for Optical Character Recognition - PyImageSearch. PyImageSearch. https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/
- Papineni, K., Roukos, S., Ward, T. J., & Zhu, W. (2002). BLEU. https://doi.org/10.3115/1073083.1073135
- Sarada, P. (2016, March). Comics as a Powerful Tool to Enhance English Language Usage. ProQuest. Retrieved February 24, 2023, from https://www.proquest.com/openview/7bea5a92132ff05d8d505aca4291e888/1?pq-origsite=gscholar 
- Sarkar, A. (2022, January 6). Understanding EfficientNet — The most powerful CNN architecture. Medium. Retrieved April 7, 2023, from https://medium.com/mlearning-ai/understanding-efficientnet-the-most-powerful-cnn-architecture-eaeb40386fad
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations. https://doi.org/10.48550/arXiv.1409.1556 
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In Proceedings of the 36th International Conference on Machine Learning (ICML).
- Vaswani, A. (2017, June 12). Attention Is All You Need. arXiv.org. https://arxiv.org/abs/1706.03762
- Veale, T., Forceville, C., & Feyaerts, K. (2014, May 23). Balloonics: The Visuals of Balloons in Comics. Academia. Retrieved October 15, 2022, from https://www.academia.edu/3012557/Balloonics_The_Visuals_of_Balloons_in_Comics
- Vothihong, P. (2017). Python: End-to-end Data Analysis. O’Reilly Online Learning. https://www.oreilly.com/library/view/python-end-to-end-data/9781788394697/ 
- Wang, C., Zhao, S., Zhu, L., Luo, K., Guo, Y., Wang, J., & Liu, S. (2021). Semi-Supervised Pixel-Level Scene Text Segmentation by Mutually Guided Network. Liushyaucheng.org. Retrieved October 1, 2022, from http://www.liushuaicheng.org/TIP/Scene_Text_Segmentation.pdf 
- Wang, Y., Liu, X., & Tang, Z. (2016, January 3). An R-CNN Based Method to Localize Speech Balloons in Comics. SpringerLink. Retrieved October 15, 2022, from https://link.springer.com/chapter/10.1007/978-3-319-27671-7_37
- Wang, Y., Wang, W., Liang, W., & Yu, L. (2019, December). Comic-Guided Speech Synthesis. Association for Computing Machinery. Retrieved October 4, 2022, from https://dl.acm.org/doi/10.1145/3355089.3356487
- Wang, Y., Zhou, Y., & Tang, Z. (2015, August). Comic frame extraction via line segments combination. 2015 13th International Conference on Document Analysis and Recognition (ICDAR). https://doi.org/10.1109/icdar.2015.7333883 
- Wang, Z., Zhang, W., Liu, J., & Wang, X. (2021). Speech Balloon Detection and Segmentation in Comics Using YOLOv5. IEEE Access, 9, 141330-141340.
- What is Cloud Storage? (2022, April 12). Google Cloud. Retrieved April 13, 2022, from https://cloud.google.com/storage/docs/introduction  
- What is CRISP-DM? (2022, March 8). Data Science Process Alliance. Retrieved March 21, 2022, from https://www.datascience-pm.com/crisp-dm-2/
- Xie, Y., & Richmond, D. (2019). Pre-training on Grayscale ImageNet Improves Medical Image Classification. Lecture Notes in Computer Science, 476–484. https://doi.org/10.1007/978-3-030-11024-6_37
- Xin, H., Ma, C., & Li, D. (2021, July 1). Comic Text Detection and Recognition Based on Deep Learning. IEEE Conference Publication | IEEE Xplore. Retrieved October 1, 2022, from https://ieeexplore.ieee.org/abstract/document/9712133?casa_token=QYLwjHn56swAAAAA:0_k_UBmzuJ-wqu7zXu8_734ivuYyDXcibbc3g8IlNsqs5__f2PahdMFifJpzZWPbEC940p-oDA 
- Yanagisawa, H., Yamashita, T., & Watanabe, H. (2018, January). A study on object detection method from manga images using CNN. 2018 International Workshop on Advanced Image Technology (IWAIT). https://doi.org/10.1109/iwait.2018.8369633
- Young-Min, K. (2019, June 25). Feature visualization in comic artist classification using deep neural networks - Journal of Big Data. SpringerOpen. Retrieved October 15, 2022, from https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0222-3 
