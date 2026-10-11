// Read-only oracle: compile with the reference visualizer geometry sources.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Web.Script.Serialization;
using SnakeRobotVisualizer;

class GeometryOracle {
    static double[] Values(object value) {
        return ((object[])value).Select(x => Convert.ToDouble(x)).ToArray();
    }
    static double[] Point(Point3D p) { return new double[]{p.X,p.Y,p.Z}; }
    static int Main(string[] args) {
        var serializer = new JavaScriptSerializer();
        var input=serializer.DeserializeObject(File.ReadAllText(args[0])) as object[];
        var output=new List<object>();
        foreach(Dictionary<string,object> fixture in input) {
            var q=Values(fixture["q"]); var lengths=Values(fixture["lengths"]);
            var profile=new RobotProfile(); profile.HeadLength=lengths[0]*1000;
            profile.Joints.Clear();
            for(int i=0;i<q.Length;i++) profile.Joints.Add(new JointConfig {
                AxisType=i%2==0?JointAxisType.Yaw:JointAxisType.Pitch,
                LinkLength=lengths[i+1]*1000 });
            var all=Kinematics.Forward(profile,q.Select((a,i)=>a*180/Math.PI*(i%2==0?1:-1)).ToArray()).Points;
            int first=Convert.ToInt32(fixture["first"]), last=Convert.ToInt32(fixture["last"]);
            var points=all.Skip(first-1).Take(last-first+4).ToArray();
            var axis=RollingAxisEstimator.Estimate(points);
            RollingProjection.SetBasis(axis,null,false);
            var projected=RollingProjection.Project(points,axis);
            var curve=CrossSectionCurve.Build(projected,Math.PI*.01);
            output.Add(new { name=fixture["name"], points=all.Select(Point).ToArray(),
                axis=Point(axis.Direction), variance=Math.Pow(axis.PitchDeviationDegrees*Math.PI/180,2),
                projected=projected.Select(p=>new double[]{p.X,p.Y}).ToArray(), valid=curve.Valid,
                segments=curve.Segments.Select(s=>new {length=s.Length,radius=s.Radius,turn=s.Turn,
                    joint=s.JointIndex,kind=s.Case}).ToArray() });
        }
        File.WriteAllText(args[1],serializer.Serialize(output)); return 0;
    }
}
