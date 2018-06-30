package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class ReportResult {

	public static void main(String[] args) throws IOException {
		double thres=0.05;
		int tp=0;
		int fp=0;
		int tn=0;
		int fn=0;
		BufferedReader br = new BufferedReader(new FileReader(new File("transduct_output.txt")));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items=line.split("\t");
			int label=(int)Double.parseDouble(items[2]);
			double predict=Double.parseDouble(items[3]);
			int predictLabel=1;
			if (predict<thres)
				predictLabel=-1;
			if (label==1 && predictLabel==1)
				tp++;
			else if (label==1 && predictLabel==-1)
				fn++;
			else if (label==-1 && predictLabel==-1)
				tn++;
			else
				fp++;
		}
		br.close();
		double pre=(double)tp/(tp+fp);
		double rec=(double)tp/(tp+fn);
		double f1=2*pre*rec/(pre+rec);
		
		System.out.println("tp: "+tp);
		System.out.println("tn: "+tn);
		System.out.println("fp: "+fp);
		System.out.println("fn: "+fn);
		System.out.println("rec: "+rec);
		System.out.println("pre: "+pre);
		System.out.println("f1: "+f1);
	}

}
