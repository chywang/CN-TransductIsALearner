package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import com.ansj.vec.Word2VEC;

import Jama.Matrix;

public class InitialProjection {

	private final static String word2vecPath="/Users/bear/Documents/workspace/Hearst/javaSkip50.model";
	private final static int dimension=50;
	private final static double lambda=0.001;
	private final static String trainPath="train.txt";
	private final static String testPath="test.txt";
	private final static String outputPath="output.txt";

	public static void main(String[] args) throws IOException {
		Word2VEC word2vec = new Word2VEC();
		word2vec.loadJavaModel(word2vecPath);
		System.out.println("word2vec load");
		Matrix trainHypoIsA=loadHypoMatrix(trainPath, word2vec, true);
		Matrix trainHyperIsA=loadHyperMatrix(trainPath, word2vec, true);
		Matrix MIsA=computeM(trainHypoIsA, trainHyperIsA, lambda);
		Matrix trainHypoNotIsA=loadHypoMatrix(trainPath, word2vec, false);
		Matrix trainHyperNotIsA=loadHyperMatrix(trainPath, word2vec, false);
		Matrix MNotIsA=computeM(trainHypoNotIsA, trainHyperNotIsA, lambda);
		predict(testPath, outputPath, MIsA, MNotIsA, word2vec);
	}
	
	@SuppressWarnings("unused")
	private static void printMatrix(Matrix m) {
		for (int i=0;i<m.getRowDimension();i++) {
			for (int j=0;j<m.getColumnDimension();j++)
				System.out.print(m.get(i, j)+" ");
			System.out.println();
		}
	}
	
	private static void predict(String testPath, String outputPath, Matrix MIsA, Matrix MNotIsA, Word2VEC word2vec) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(testPath)));
		PrintWriter pw=new PrintWriter(outputPath);
		String line;
		while ((line = br.readLine()) != null) {
			String[] items=line.split("\t");
			String hypo=items[0];
			String hyper=items[1];
			String label=items[2];
			pw.println(hypo+"\t"+hyper+"\t"+label+"\t"+computeScore(hypo, hyper, MIsA, MNotIsA, word2vec)+"\t"+computeConf(hypo, hyper, MIsA, MNotIsA, word2vec));
			pw.flush();
		}
		br.close();
		pw.close();
	}
	
	private static double computeScore(String hypo, String hyper, Matrix MIsA, Matrix MNotIsA, Word2VEC word2vec) {
		float[] hf1=word2vec.getWordVector(hypo);
		double[][] ds1=new double[hf1.length][1];
		for (int i=0;i<hf1.length;i++)
			ds1[i][0]=hf1[i];
		Matrix hypoM=new Matrix(ds1);
		float[] hf2=word2vec.getWordVector(hyper);
		double[][] ds2=new double[hf2.length][1];
		for (int i=0;i<hf2.length;i++)
			ds2[i][0]=hf2[i];
		Matrix hyperM=new Matrix(ds2);
		double positiveScore=MIsA.times(hypoM).minus(hyperM).normF();
		double negativeScore=MNotIsA.times(hypoM).minus(hyperM).normF();
		return Math.tanh(negativeScore-positiveScore);
	}
	
	private static double computeConf(String hypo, String hyper, Matrix MIsA, Matrix MNotIsA, Word2VEC word2vec) {
		float[] hf1=word2vec.getWordVector(hypo);
		double[][] ds1=new double[hf1.length][1];
		for (int i=0;i<hf1.length;i++)
			ds1[i][0]=hf1[i];
		Matrix hypoM=new Matrix(ds1);
		float[] hf2=word2vec.getWordVector(hyper);
		double[][] ds2=new double[hf2.length][1];
		for (int i=0;i<hf2.length;i++)
			ds2[i][0]=hf2[i];
		Matrix hyperM=new Matrix(ds2);
		double positiveScore=MIsA.times(hypoM).minus(hyperM).normF();
		double negativeScore=MNotIsA.times(hypoM).minus(hyperM).normF();
		return Math.abs(negativeScore-positiveScore)/Math.max(positiveScore, negativeScore);
	}
	
	private static Matrix computeM(Matrix xMatrix, Matrix yMatrix, double lambda) {
	//	printMatrix(xMatrix);
	//	printMatrix(yMatrix);
		double[][] d=new double[dimension][dimension];
		for (int i=0;i<d.length;i++)
			for (int j=0;j<d[i].length;j++) {
				if (i==j)
					d[i][j]=1;
				else
					d[i][j]=0;
			}
		Matrix identity=new Matrix(d);
		return yMatrix.times(xMatrix.transpose()).times((xMatrix.times(xMatrix.transpose()).plus(identity.times(lambda))).inverse());
	}
	
	private static Matrix loadHypoMatrix(String path, Word2VEC word2vec, boolean isPositive) throws IOException {
		List<String> words=new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(new File(path)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items=line.split("\t");
			String hypo=items[0];
			String label=items[2];
			if (isPositive) {
				if (label.equals("1"))
					words.add(hypo);
			} else {
				if (label.equals("0"))
					words.add(hypo);
			}
		}
		br.close();
	
		double[][] ds1=new double[dimension][words.size()];
		for (int m=0;m<words.size();m++) {
			float[] hf1=word2vec.getWordVector(words.get(m));
			for (int i=0;i<hf1.length;i++)
				ds1[i][m]=hf1[i];
		}
		br.close();
		return new Matrix(ds1);
	}
	
	private static Matrix loadHyperMatrix(String path, Word2VEC word2vec, boolean isPositive) throws IOException {
		List<String> words=new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(new File(path)));
		String line;
		while ((line = br.readLine()) != null) {
			String[] items=line.split("\t");
			String hyper=items[1];
			String label=items[2];
			if (isPositive) {
				if (label.equals("1"))
					words.add(hyper);
			} else {
				if (label.equals("0"))
					words.add(hyper);
			}
		}
		br.close();
		double[][] ds1=new double[dimension][words.size()];
		for (int m=0;m<words.size();m++) {
			float[] hf1=word2vec.getWordVector(words.get(m));
			for (int i=0;i<hf1.length;i++)
				ds1[i][m]=hf1[i];
		}
		br.close();
		return new Matrix(ds1);
	}

}
