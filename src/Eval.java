import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Eval {

	public static void main(String[] args) throws IOException {
		if (args.length != 3) {
			System.out.println("argments input error!");
			return;
		}

		String truthPath = args[0];
		String predictPath = args[1];
		double thres = Double.parseDouble(args[2]); // set to 0.05 - 0.1 as
													// default

		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;

		Map<String, Integer> truthMap = new HashMap<String, Integer>();
		BufferedReader br = new BufferedReader(new FileReader(new File(truthPath)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hypo = items[0];
			String hyper = items[1];
			int label = (int) Double.parseDouble(items[2]);
			truthMap.put(hypo + " " + hyper, label);
		}
		br.close();

		br = new BufferedReader(new FileReader(new File(predictPath)));
		while ((line = br.readLine()) != null) {
			String[] items = line.split("\t");
			String hypo = items[0];
			String hyper = items[1];
			int label = truthMap.get(hypo + " " + hyper);
			if (label == 0)
				label = -1;
			double predict = Double.parseDouble(items[2]);
			int predictLabel = 1;
			if (predict < thres)
				predictLabel = -1;
			if (label == 1 && predictLabel == 1)
				tp++;
			else if (label == 1 && predictLabel == -1)
				fn++;
			else if (label == -1 && predictLabel == -1)
				tn++;
			else
				fp++;
		}
		br.close();
		double pre = (double) tp / (tp + fp);
		double rec = (double) tp / (tp + fn);
		double f1 = 2 * pre * rec / (pre + rec);

		System.out.println("tp: " + tp);
		System.out.println("tn: " + tn);
		System.out.println("fp: " + fp);
		System.out.println("fn: " + fn);
		System.out.println("rec: " + rec);
		System.out.println("pre: " + pre);
		System.out.println("f1: " + f1);
	}

}
