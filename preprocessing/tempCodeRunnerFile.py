                    x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                    y = np.arctan2(-rmat[2, 0], sy)
                    z = 0

                # Convert to degrees
                pitch = np.degrees(x)
                yaw = np.degrees(y)
                roll = np.degrees(z)

                # CHANGED: Visualize the Yaw for the user
                info = f"Yaw: {int(yaw)} deg"
                cv2.putText(image, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 255), 2)
                
                # OPTIONAL: Draw the axis (Nose pointer)
                nose_2d = (int(face_landmarks.landmark[1].x * img_w), 
                           int(face_landmarks.landmark[1].y * img_h))
                
                # Project a 3D point for the nose direction
                p1 = (int(nose_2d[0] + y * 1000), int(nose_2d[1])) # Project yaw direction
                # This is a simplification; for real axis projection use projectPoints
                
        return image
testing= FaceRotationFilter()
