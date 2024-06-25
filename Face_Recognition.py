import cv2
import face_recognition

# Empty list of faces encodings & names below
known_face_encodings = []
known_face_names = []


#We create a known_person_image variable & call the load_image_file function which takes a jpg
known_person1_image = face_recognition.load_image_file("person1.jpg")
known_person2_image = face_recognition.load_image_file("person2.jpg")
known_person3_image = face_recognition.load_image_file("person3.jpg")


#Now that we have the image of the person saved, we declare an encoding variable & asssign the image to that
known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]


#Adding each encoding to the known_face_encodings list
known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)
known_face_encodings.append(known_person3_encoding)


#Adding each encoding to the known_face_names list
known_face_names.append("Edred Azziz")
known_face_names.append("Barack Obama")
known_face_names.append("Margot Robbie")



# Initialize webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 30)  # Set capture FPS to 30
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width of the frames
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height of the frames

while True:
    # Capture each frame
    ret, frame = video_capture.read()

    # Find all face LOCATIONS in current frame. Finding locations allows for the program to be more efficent rather than simply analyzing entire frame.
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches ANY known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default to "Unknown" unless a match is found

        if True in matches: #If a match is found, we find WHICh match has been found.
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            # Draw a green rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Draw a red rectangle around the face for unknown faces.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting image in the window
    cv2.imshow("Video", frame)

    # Break the loop of facial recog when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam & close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
