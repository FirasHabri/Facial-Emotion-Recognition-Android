package com.example.s_oss.facedetector;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Display;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

//Tensorflow libs for DeepLearning
import org.tensorflow.lite.Interpreter;

//OpenCV libs for ImageProcessing
import org.opencv.android.OpenCVLoader;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.android.Utils;
import org.opencv.core.MatOfByte;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

import static java.sql.DriverManager.println;
import static org.opencv.core.CvType.CV_32F;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    TextView testView;

    Camera camera;
    SurfaceView surfaceView;
    SurfaceHolder surfaceHolder;

    Camera.PictureCallback rawCallback;
    Camera.ShutterCallback shutterCallback;
    Camera.PictureCallback jpegCallback;

    private Interpreter tflite;

    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        final Activity activity = this;
        setContentView(R.layout.activity_main);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "OpenCV loaded Successfully", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(getApplicationContext(), "Could not Load OpenCV", Toast.LENGTH_SHORT).show();
        }


        surfaceView = (SurfaceView) findViewById(R.id.surfaceView);
        surfaceHolder = surfaceView.getHolder();

        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        surfaceHolder.addCallback(this);

        // deprecated setting, but required on Android versions prior to 3.0
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        jpegCallback = new Camera.PictureCallback() {
            public void onPictureTaken(byte[] data, Camera camera) {
                List<String> emotion_labels = Arrays.asList("angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral");
                FileOutputStream outStream = null;
                try {
                    String root = Environment.getExternalStorageDirectory().toString();
                    String filepath = String.format(root + String.format("/%d.jpg", System.currentTimeMillis()));
                    //outStream = new FileOutputStream(filepath);

                    //get the camera parameters
                    Camera.Parameters parameters = camera.getParameters();
                    int width = parameters.getPreviewSize().width;
                    int height = parameters.getPreviewSize().height;

                    Mat mat = Imgcodecs.imdecode(new MatOfByte(data), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
                    Mat rotationMatrix = Imgproc.getRotationMatrix2D(new Point(mat.rows() / 2, mat.cols() / 2), 90, 1);
                    Imgproc.warpAffine(mat, mat, rotationMatrix, new Size(mat.cols(), mat.cols()));

                    File path = new File(Environment.getExternalStorageDirectory() + "/Images/");
                    path.mkdirs();
                    File file = new File(path, "image.png");

                    String filename = file.toString();

                    Mat format_im = format_image(mat);

                    //float array2d[][] = Mat_to_2dArray(format_im);
                    float[][][][] reshape = reshape(format_im);

                    float[] returnedArray = new float[7];
                    float[][] arrayofreturnedArray = {returnedArray};

                    tflite.run(reshape, arrayofreturnedArray);

                    float maxVal = findMaxValue(arrayofreturnedArray[0]);
                    int maxInd = findMaxIndex(arrayofreturnedArray[0]);

                    String emotionText = emotion_labels.get(maxInd);

                    //Boolean bool = Imgcodecs.imwrite(filename, output);



                        TextView textview = (TextView) findViewById(R.id.textView);
                        textview.setText("Detected emotion: " + emotionText);


                }catch (Exception e) {
                    e.printStackTrace();
                }
                refreshCamera();
            }
        };
    }

    public void captureImage(View v) throws IOException {
        //take the picture
        camera.takePicture(null, null, jpegCallback);
    }

    public void refreshCamera() {
        if (surfaceHolder.getSurface() == null) {
            // preview surface does not exist
            return;
        }

        // stop preview before making changes
        try {
            camera.stopPreview();
        } catch (Exception e) {
            // ignore: tried to stop a non-existent preview
        }

        // set preview size and make any resize, rotate or
        // reformatting changes here
        // start preview with new settings
        try {
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
        } catch (Exception e) {

        }
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {

        Camera.Parameters parameters = camera.getParameters();
        Display display = ((WindowManager) getSystemService(WINDOW_SERVICE)).getDefaultDisplay();

        if (display.getRotation() == Surface.ROTATION_0) {

            camera.setDisplayOrientation(90);
        }

        if (display.getRotation() == Surface.ROTATION_90) {

        }

        if (display.getRotation() == Surface.ROTATION_180) {

        }

        if (display.getRotation() == Surface.ROTATION_270) {

            camera.setDisplayOrientation(180);
        }

        camera.setParameters(parameters);
        refreshCamera();
    }

    private int FFC() {
        int cameraCount = 0;
        int cam = 0;
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        cameraCount = Camera.getNumberOfCameras();
        for (int camIdx = 0; camIdx < cameraCount; camIdx++) {
            Camera.getCameraInfo(camIdx, cameraInfo);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                try {
                    cam = camIdx;
                } catch (RuntimeException e) {
                    Log.e("err: ", "Camera failed to open: " + e.getLocalizedMessage());
                }
            }
        }

        return cam;
    }

    public void surfaceCreated(SurfaceHolder holder) {
        try {
            // open the camera
            camera = Camera.open(FFC());
        } catch (RuntimeException e) {
            // check for exceptions
            System.err.println(e);
            return;
        }
        Camera.Parameters param;
        param = camera.getParameters();

        // modify parameter
        param.setPreviewSize(352, 288);
        camera.setParameters(param);
        try {
            // The Surface has been created, now tell the camera where to draw
            // the preview.
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
        } catch (Exception e) {
            // check for exceptions
            System.err.println(e);
            return;
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // stop preview and release camera
        camera.stopPreview();
        camera.release();
        camera = null;
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private Mat format_image(Mat image) {

        // load cascade file from application resources
        File mCascadeFile = null;
        try {
            InputStream is = getResources().getAssets().open("haarcascade_frontalface_default.xml");
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        CascadeClassifier cascade_classifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        cascade_classifier.load(mCascadeFile.getAbsolutePath());
        if (cascade_classifier.empty()) {
            Toast.makeText(getApplicationContext(), "Can not find haarcascade file", Toast.LENGTH_SHORT).show();
        }
        if (image.channels() > 2) {
            Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY);
        } else {

        }
        MatOfRect faceVectors = new MatOfRect();
        cascade_classifier.detectMultiScale(image, faceVectors, 1.1, 2);
        Rect[] faces = faceVectors.toArray();
        Rect[] max_area_face = faces;
        for (Rect face : faces) {
            System.out.println(face);
            if (face.width * face.height > max_area_face[0].height * max_area_face[0].width) {
                max_area_face[0] = face;
            }
        }
        Rect face = max_area_face[0];

        for (Rect rect : faceVectors.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
            face = new Rect(rect.x, rect.y, rect.width, rect.height);
        }
        Mat image_roi = new Mat(image, face);
        try {
            int scaledWidth = 64;
            int scaledHeight = 64;
            Size sz = new Size(scaledWidth, scaledHeight);
            int interpolation = Imgproc.INTER_CUBIC;

            Imgproc.resize(image_roi, image_roi, sz, 0, 0, interpolation);
            image_roi.convertTo(image_roi, CvType.CV_32F, 1.0 / 255, 0);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return image_roi;
    }

    private static float[][][][] reshape(Mat mArray) {

        float[][][][] resultArray = new float[1][mArray.cols()][mArray.rows()][1];

        for (int i = 0; i < mArray.cols(); i++) {
            for (int j = 0; j < mArray.rows(); j++) {
                resultArray[0][i][j][0] = (float) mArray.get(i, j)[0];// mArray[i][j];
            }
        }

        return resultArray;
    }

    private float[][] Mat_to_2dArray(Mat image) {
        int size = (int) image.total() * image.channels();
        float[] data = new float[size];
        image.put(0, 0, data);
        float array2d[][] = new float[64][64];

        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                array2d[i][j] = data[(j * 10) + i];
            }
        }
        return array2d;
    }

    private float findMaxValue(float[] array) {
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
    }

    private int findMaxIndex(float[] array) {
        float max = array[0];
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }
}
