using System;
using System.Collections.Generic;
using ImageDetectionUtility;
using OpenCvSharp;

namespace GrundrissTester
{
    class Program
    {
        static void Main(string[] args)
        {
            test1();
            test2();
        }
        static void test1()
        {
            Mat input = new Mat("input.jpg", ImreadModes.GrayScale);
            Mat SW = IDU.makeSW(input);
            using (new Window("SW", WindowMode.AutoSize, SW))
            {
                Window.WaitKey(0);
            }
            Mat oedges = new Mat(input.Size(), input.Type());
            List<int[]> cornerlist = new List<int[]>();
            IDU.filterimage(ref SW, ref oedges, ref cornerlist);
            using (new Window("oedges", WindowMode.AutoSize, oedges))
            using (new Window("filtered", WindowMode.AutoSize, SW))
            {
                Window.WaitKey(0);
            }
            SW.SaveImage("filtered.png");
            List<int[]> cornerlist2 = new List<int[]>();
            Mat cornersmarked = IDU.getcorners(SW, ref oedges, ref cornerlist2);
            using (new Window("oedges2", WindowMode.AutoSize, oedges))
            using (new Window("cornersmarked", WindowMode.AutoSize, cornersmarked))
            {
                Window.WaitKey(0);
            }
            Mat ImageOut = new Mat(SW.Size(), SW.Type());
            List<int[]> reducedcornerlist = IDU.reducecornerlist(cornerlist2, 5);
            List<int[]> linelist = IDU.getlines(SW, reducedcornerlist, 92);
            ImageOut.SetTo(new Scalar(255));
            ImageOut = IDU.drawLineArray(ImageOut, linelist);
            using (new Window("ImageOut", WindowMode.AutoSize, ImageOut))
            {
                Window.WaitKey(0);
            }
            ImageOut.SaveImage("final.png");
            Console.WriteLine(Cv2.GetCudaEnabledDeviceCount() + " Cuda fähiges Gerät erkannt!");
            Console.ReadLine();
        }
        static void test2()
        {
            Mat input = new Mat("input.jpg", ImreadModes.GrayScale);
            List<int[]> linelist = IDU.ProcessImage(input, 5, 92);
            foreach(int[] element in linelist)
            { Console.WriteLine("P1("+element[0]+", "+element[1]+") ---> P2("+element[2]+", "+element[3]+")"); }
            Console.ReadLine();
        }
    }
}