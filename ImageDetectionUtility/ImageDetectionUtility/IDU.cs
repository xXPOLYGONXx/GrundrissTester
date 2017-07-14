using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;

namespace ImageDetectionUtility
{
    /// <summary>
    /// Diese Klasse stellt die Funktionen zur Erfassung von Grundrissen zur Verfügung
    /// </summary>
    public class IDU
    {
        //____________________________________________________________________________________
        /// <summary>
        /// Errechnet die Ecken in einem Bild und speichert diese in der Cornerlist.
        /// Weiterhin wird ein Bild erstellt auf dem die gefundenen Ecken eingezeichnetsind (src)
        /// im onlyedges Bild sind nur die Ecken markiert.
        /// </summary>
        static public Mat getcorners(Mat src, ref Mat onlyedges, ref List<int[]> cornerlist)
        {
            Mat edges = new Mat();
            edges = getEdges(src, 50);
            //new Window("Edges for Cornerdetection", edges);
            // Corner detection
            // Get All Processing Images
            Mat cross = createACross();
            Mat diamond = createADiamond();
            Mat square = createASquare();
            Mat x = createAXShape();
            Mat dst = new Mat();

            // Dilate with a cross
            Cv2.Dilate(src, dst, cross);

            // Erode with a diamond
            Cv2.Erode(dst, dst, diamond);

            Mat dst2 = new Mat();

            // Dilate with a X
            Cv2.Dilate(src, dst2, x);

            // Erode with a square
            Cv2.Erode(dst2, dst2, square);

            // Corners are obtain by differencing the two closed images
            Cv2.Absdiff(dst, dst2, dst);
            applyThreshold(dst, 45);
            onlyedges = dst;

            // The following code Identifies the founded corners by
            // drawing circle on the src image.
            cornerlist = IDTheCorners(dst, src, cornerlist);
            return src;
         }
        //____________________________________________________________________________________
        private static Mat createACross()
        {
            Mat cross = new Mat(5, 5, MatType.CV_8U, new Scalar(0));

            // creating the cross-shaped structuring element
            for (int i = 0; i < 5; i++)
            {
                cross.Set<byte>(2, i, 1);
                cross.Set<byte>(i, 2, 1);
            }

            return cross;
        }
        //____________________________________________________________________________________
        private static Mat createADiamond()
        {
            Mat diamond = new Mat(5, 5, MatType.CV_8U, new Scalar(1));

            // Creating the diamond-shaped structuring element
            diamond.Set<byte>(0, 0, 0);
            diamond.Set<byte>(1, 0, 0);
            diamond.Set<byte>(3, 0, 0);
            diamond.Set<byte>(4, 0, 0);
            diamond.Set<byte>(0, 1, 0);
            diamond.Set<byte>(4, 1, 0);
            diamond.Set<byte>(0, 3, 0);
            diamond.Set<byte>(4, 3, 0);
            diamond.Set<byte>(4, 4, 0);
            diamond.Set<byte>(0, 4, 0);
            diamond.Set<byte>(1, 4, 0);
            diamond.Set<byte>(3, 4, 0);

            return diamond;
        }
        //____________________________________________________________________________________
        private static Mat createASquare()
        {
            Mat Square = new Mat(5, 5, MatType.CV_8U, new Scalar(1));

            return Square;
        }
        //____________________________________________________________________________________
        private static Mat createAXShape()
        {
            Mat x = new Mat(5, 5, MatType.CV_8U, new Scalar(0));

            // Creating the x-shaped structuring element
            for (int i = 0; i < 5; i++)
            {
                x.Set<byte>(i, i, 1);
                x.Set<byte>(4 - i, i, 1);
            }

            return x;
        }
        //____________________________________________________________________________________
        private static void applyThreshold(Mat result, int threshold)
        {
            Cv2.Threshold(result, result, threshold, 255, ThresholdTypes.Binary);
        }
        //____________________________________________________________________________________
        /// <summary>
        /// Diese Funktion Umkreist alle Ecken aus dem Bild "binary" im Bild "image"
        /// binary ist ein bild auf dem nur die Ecken als Punkte eingezeichnet sind
        /// Die Ecken werden ebenfalls in der Cornerlist abgespeichert
        /// </summary>
        public static List<int[]> IDTheCorners(Mat binary, Mat image, List<int[]> cornerlist)
        {
            for (int r = 0; r < binary.Rows; r++)
                for (int c = 0; c < binary.Cols; c++)
                    if (binary.At<byte>(r, c) != 0)
                    {
                        Cv2.Circle(image, c, r, 5, new Scalar(0, 0, 255));
                        cornerlist.Add(new int[2] { c, r });
                    }
            return cornerlist;
        }
        //____________________________________________________________________________________
        private static Mat getEdges(Mat image, int threshold)
        {
            // Get the gradient image
            Mat result = new Mat();
            Cv2.MorphologyEx(image, result, MorphTypes.Gradient, new Mat());
            applyThreshold(result, threshold);

            return result;
        }
        //____________________________________________________________________________________
        /// <summary>
        /// Bestimmt die Linien im Eingangsbild anhand der Eckenliste und gibt Eine Liste der gefundenen Linien zurück.
        /// Läuft sehr rechenintensiv mit Multithreading!!!
        /// Der Parameter blackpercentage gibt an ab welchem prozentualen Schwarzanteil eine Wand zwischen zwei Punkten gezogen wird.
        /// Empfohlene Werte 80-95
        /// </summary>
        public static List<int[]> getlines(Mat imageIn, List<int[]> cornerlist, double blackpercentage)
        {
            imageIn.SaveImage("temp.png");
            Mat origimage = new Mat("temp.png",ImreadModes.Color);
            origimage = imageIn;
            List<int[]> linelist = new List<int[]>();
            int counter = 0;
            List<int[]> cornerlist1 = cornerlist;
            List<int[]> cornerlist2 = cornerlist;
            Parallel.ForEach(
                cornerlist1,
                new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount * 2 },
                corner => { getlinesmt(origimage, cornerlist2, corner, ref linelist, blackpercentage);counter++; Console.Clear(); Console.WriteLine(counter + " Threads von: " + cornerlist.Count); }
            );
           /* List<int[]> reducedlinelist = new List<int[]>();
            foreach(int[] line in linelist)
            {
                if(reducedlinelist.Contains(line))
                { }
                else
                {
                    reducedlinelist.Add(line);
                }
            }*/
            return linelist;
        }
        //____________________________________________________________________________________
        private static void getlinesmt(Mat origimage, List<int[]> cornerlist2, int[] corner, ref List<int[]> linelist,double blackpercentage)
        {          
            foreach (int[] secondcorner in cornerlist2)
            {
                int[] line = new int[4];
                int counter = 0;
                if (corner != secondcorner)
                {
                    List<int[]> linepoints = GetLinePoints(corner[0], corner[1], secondcorner[0], secondcorner[1]);
                    foreach (int[] pixel in linepoints)
                    {
                        Vec3b pix1 = origimage.At<Vec3b>(pixel[1], pixel[0]);
                        Vec3b pix2 = origimage.At<Vec3b>(pixel[1] + 1, pixel[0] + 1);
                        Vec3b pix3 = origimage.At<Vec3b>(pixel[1] - 1, pixel[0] - 1);
                        Vec3b pix4 = origimage.At<Vec3b>(pixel[1] + 2, pixel[0] + 2);
                        Vec3b pix5 = origimage.At<Vec3b>(pixel[1] + 3, pixel[0] + 3);
                        Vec3b pix6 = origimage.At<Vec3b>(pixel[1] - 2, pixel[0] - 2);
                        if ((pix1.Item0 < 20 && pix1.Item1 < 20 && pix1.Item2 < 20) || (pix2.Item0 < 20 && pix2.Item1 < 20 && pix2.Item2 < 20) || (pix3.Item0 < 20 && pix3.Item1 < 20 && pix3.Item2 < 20) || (pix4.Item0 < 20 && pix4.Item1 < 20 && pix4.Item2 < 20) || (pix5.Item0 < 20 && pix5.Item1 < 20 && pix5.Item2 < 20) || (pix6.Item0 < 20 && pix6.Item1 < 20 && pix6.Item2 < 20))
                        {
                            counter++;
                        }
                    }
                    if (counter > linepoints.Count * (blackpercentage/100))
                    //if (counter > (linepoints.Count * 0.97))
                    {
                        //imageIn.Line(corner[0], corner[1], secondcorner[0], secondcorner[1], new Scalar(0, 0, 0), 1);
                        line[0] = corner[0];
                        line[1] = corner[1];
                        line[2] = secondcorner[0];
                        line[3] = secondcorner[1];
                        Console.WriteLine("P1(" + line[0] + ", " + line[1] + ") ---> P2(" + line[2] + ", " + line[3] + ")");
                        linelist.Add(line);
                        
                    }
                    counter = 0;
                }
            }
        }
        //____________________________________________________________________________________
        private static List<int[]> GetLinePoints(int x1, int y1, int x2, int y2)
        {
            double steigung = 0.0;
            steigung = (double)(y1 - y2) / (double)(x1 - x2);
            List<int[]> punktliste = new List<int[]>();
            double laenge = Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));
            if (laenge != 0)
            {
                int[] temp;
                for (int i = 0; i < (int)laenge; i++)
                {
                    temp = new int[2];
                    int x = (int)(Math.Round(x1 + i * (x2 - x1) / laenge, 2));
                    int y = (int)(Math.Round(y1 + i * (y2 - y1) / laenge, 2));
                    temp[0] = x;
                    temp[1] = y;
                    punktliste.Add(temp);

                }
                if (punktliste.Last()[0] != x2 && punktliste.Last()[1] != y2)
                {
                    temp = new int[2];
                    temp[0] = x2;
                    temp[1] = y2;
                    punktliste.Add(temp);
                }
            }
            return punktliste;
        }
        //____________________________________________________________________________________
        /// <summary>
        /// verwandelt das Eingangsbild in ein Bild mit den Farbwerten 0 und 255
        /// </summary>
        public static Mat makeSW(Mat input)
        {
            var indexer = input.GetGenericIndexer<byte>();
                for (int y = 0; y < input.Height; y++)
                {
                    for (int x = 0; x < input.Width; x++)
                    {
                        byte color = indexer[y, x];
                        if (color > 170)
                        {
                            color = 255;
                            
                        }
                        else
                        {
                            color = 0;
                            
                        }
                        indexer[y, x] = color;
                    }
                }
                return input;
        }
        //____________________________________________________________________________________
        /// <summary>
        /// zeichnet die Linien aus der Liste in das übergebene Bild und gibt dieses dann zurück
        /// </summary>
        public static Mat drawLineArray(Mat input, List<int[]> linelist)
        {
            //linelist vom Typ List<int[4]> mit [0] gleich x1 und [2] gleich x2
            foreach (int[] element in linelist)
            {
                OpenCvSharp.Cv2.Line(input, element[0], element[1], element[2], element[3], new OpenCvSharp.Scalar(0), 1);
            }
            return input;
        }
        //____________________________________________________________________________________
        /// <summary>
        /// reduziert die eingegebene Eckenliste indem Gruppierungen zusammengefasst werden
        /// Der Parameter Cornerdistance gibt an ab welcher Distanz Ecken zusammengefasst werden.
        /// Wert 0-5 --> je nach Auflösung sehr genaue Wandfindung (Performance sinkt)
        /// Wert 5-20 --> je nach Auflösung relativ ungenaue Wandfindung (Performance steigt drastisch)
        /// </summary>
        public static List<int[]> reducecornerlist(List<int[]> cornerlist, int Cornerdistance)
        {
            List<int[]> reducedcornerlist = new List<int[]>();
            Boolean add = true;
            int tempx = 0;
            int tempy = 0;
            foreach (int[] element in cornerlist)
            {
                if (reducedcornerlist.Count == 0)
                {
                    reducedcornerlist.Add(new int[2] { element[0], element[1] });
                }
                else
                {
                    foreach (int[] corner in reducedcornerlist)
                    {
                        tempx = element[0] - corner[0];
                        tempy = element[1] - corner[1];
                        if (tempx < Cornerdistance && tempx > (-1*Cornerdistance) && tempy < Cornerdistance && tempy > (-1 * Cornerdistance))
                        {
                            corner[0] = (corner[0] + element[0]) / 2;
                            corner[1] = (corner[1] + element[1]) / 2;
                            add = false;
                        }
                    }
                    if (add == true)
                    {
                        reducedcornerlist.Add(new int[2] { element[0], element[1] });
                    }
                    add = true;
                }

            }
            return reducedcornerlist;
        }
        //____________________________________________________________________________________
        /// <summary>
        /// Filter den Grundriss auf relevante Details. Ausgegeben wird das Ergebnis im Bild src
        /// in onlyedges finden sich nur als relevant befundene Punkte (invertiertes Ergebnis)
        /// </summary>
        static public void filterimage(ref Mat src, ref Mat onlyedges, ref List<int[]> cornerlist)
        {
            Mat edges = new Mat();
            // Show Edges
            edges = getEdges(src, 50);
            //new Window("Edges for Cornerdetection", edges);
            // Corner detection
            // Get All Processing Images
            Mat cross = createACross();
            Mat diamond = createADiamond();
            Mat square = createASquare();
            Mat x = createAXShape();
            Mat t = createAT();
            Mat dragon = createADragon();

            Mat dst = new Mat();

            // Dilate with a cross
            Cv2.Dilate(src, dst, cross);

            // Erode with a diamond
            Cv2.Erode(dst, dst, diamond);

            Mat dst2 = new Mat();

            // Dilate with a X
            Cv2.Dilate(src, dst2, x);

            // Erode with a square
            Cv2.Erode(dst2, dst2, square);

            Mat dst3 = new Mat();

            // Dilate with a T
            Cv2.Dilate(src, dst3, t);

            // Erode with a dragon
            Cv2.Erode(dst3, dst3, dragon);

            // Corners are obtain by differencing the two closed images
            //__________________________________________________________________________________________________________
            Cv2.Absdiff(dst, dst3, dst);
            //Cv2.Absdiff(dst, dst2, dst);
            applyThreshold(dst, 45);
            onlyedges = dst;
            //__________________________________________________________________________________________________________

            // The following code Identifies the founded corners by
            // drawing circle on the src image.

            //IDTheCorners(dst, src, ref cornerlist);
            src = Blackpoints(dst, src, ref cornerlist);
        }
        //____________________________________________________________________________________
        static Mat createAT()
        {
            Mat T = new Mat(5, 5, MatType.CV_8U, new Scalar(1));

            // Creating the T-shaped structuring element
            T.Set<byte>(1, 1, 0);
            T.Set<byte>(1, 2, 0);
            T.Set<byte>(1, 3, 0);
            T.Set<byte>(2, 2, 0);
            T.Set<byte>(3, 2, 0);

            return T;
        }
        //____________________________________________________________________________________
        private static Mat Blackpoints(Mat binary, Mat image, ref List<int[]> cornerlist)
        {
            Mat image2 = new Mat(image.Size(), image.Type());
            image2.SetTo(new Scalar(255));
            for (int r = 0; r < binary.Rows; r++)
                for (int c = 0; c < binary.Cols; c++)
                    if (binary.At<byte>(r, c) != 0)
                    {
                        image2.Set<byte>(r, c, 0);
                        image2.Set<byte>(r+1, c+1, 0);
                        image2.Set<byte>(r+1, c, 0);
                        image2.Set<byte>(r+1, c-1, 0);
                        image2.Set<byte>(r, c+1, 0);
                        image2.Set<byte>(r, c-1, 0);
                        image2.Set<byte>(r-1, c+1, 0);
                        image2.Set<byte>(r-1, c, 0);
                        image2.Set<byte>(r-1, c-1, 0);
                        Cv2.Circle(image2, c, r, 1, new Scalar(0, 0, 255));
                        cornerlist.Add(new int[2] { c, r });
                    }
            return image2;
        }
        //____________________________________________________________________________________
        static Mat createADragon()
        {
            Mat Dragon = new Mat(5, 5, MatType.CV_8U, new Scalar(1));

            // Creating the reverseT-shaped structuring element
            for (int i = 0; i < 5; i++)
            {
                Dragon.Set<byte>(0, i, 0);
                Dragon.Set<byte>(4, i, 0);
                Dragon.Set<byte>(i, 0, 0);
                Dragon.Set<byte>(i, 4, 0);
            }
            Dragon.Set<byte>(3, 1, 0);
            Dragon.Set<byte>(3, 3, 0);

            return Dragon;
        }
        //____________________________________________________________________________________
        /// <summary>
        /// Filter den Grundriss auf relevante Details. Und erkennt die Wände.
        /// Als Ergebnis bekommt man eine Liste mit allen gefundenen Wänden
        /// Uinput ist das EIngangsbild
        /// Der Parameter Cornerdistance gibt an ab welcher Distanz Ecken zusammengefasst werden.
        /// Wert 0-5 --> je nach Auflösung sehr genaue Wandfindung (Performance sinkt)
        /// Wert 5-20 --> je nach Auflösung relativ ungenaue Wandfindung (Performance steigt drastisch)
        /// 
        /// Der Parameter blackpercentage gibt an ab welchem prozentualen Schwarzanteil eine Wand zwischen zwei Punkten gezogen wird.
        /// Empfohlene Werte 80-95
        /// </summary>
        public static List<int[]> ProcessImage(Mat Uinput,int cornerdistance, double blackpercentage)
        {
            Uinput.SaveImage("temp.png");
            Mat input = new Mat("temp.png", ImreadModes.GrayScale);
            System.IO.File.Delete("temp.png");
            Mat SW = IDU.makeSW(input);
            Mat oedges = new Mat(input.Size(), input.Type());
            List<int[]> cornerlist = new List<int[]>();
            IDU.filterimage(ref SW, ref oedges, ref cornerlist);
            List<int[]> cornerlist2 = new List<int[]>();
            Mat cornersmarked = IDU.getcorners(SW, ref oedges, ref cornerlist2);
            List<int[]> reducedcornerlist = IDU.reducecornerlist(cornerlist2, cornerdistance);
            List<int[]> linelist = IDU.getlines(SW, reducedcornerlist,blackpercentage);
            return linelist;
        }
    }
}
