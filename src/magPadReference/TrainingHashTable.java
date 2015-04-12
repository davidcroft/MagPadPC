package magPadReference;

import java.util.ArrayList;
import java.util.HashMap;

public class TrainingHashTable {
	
	// Hashtable
	//Pair<Float, Float> pair = Pair.createPair(1.0, 2.0);
	public final ArrayList<HashMap<Integer, Pair<Float, Float>>> htReg;
	public final HashMap<Integer, String> htCls = new HashMap<Integer, String>();
	public final int[] trainRegTotalNum;
	
	TrainingHashTable() {
		trainRegTotalNum = new int[GlobalConstants.TRAININGCLSNUM];
		htReg =  new ArrayList<HashMap<Integer, Pair<Float, Float>>>();
		// regression table
		for (int i = 0; i < GlobalConstants.TRAININGCLSNUM; i++) {
			HashMap<Integer, Pair<Float, Float>> htMap = new HashMap<Integer, Pair<Float, Float>>();
			int index = 1;
			float xOffset = (float) ((i%2) == 0? 0.6 : 5.4);
			float yOffset = (float) (i<2? 0.6 : 3.6);
			int colSum = i<2? GlobalConstants.TRAINTOTALCOL : GlobalConstants.TRAINTOTALCOL+1;
			int rowSum = (i%2 == 0)? GlobalConstants.TRAINTOTALROW : GlobalConstants.TRAINTOTALROW-1;
			trainRegTotalNum[i] = colSum*rowSum;
			for (int col = 0; col < colSum; col++) {
				for (int row = 0; row < rowSum; row++) {
					float x = (float) ((0.4 * row + xOffset) / GlobalConstants.MAXHEIGHT);
					float y = (float) ((0.6 * col + yOffset) / GlobalConstants.MAXWIDTH);
					htMap.put(index++, Pair.createPair(x, y));
				}
			}
			htReg.add(htMap);
		}
		
		int index = 1;
		for (int i = 0; i < GlobalConstants.TRAININGCLSNUM+1; i++) {
			htCls.put(index++, "block"+i);
		}
		
		System.out.println("Create " + (htReg.size()+1) + " hashtables for regression and classification");
	}
	
	Pair<Float, Float> getReg(int index, int key) {
		if (index < htReg.size() && htReg.get(index).containsKey(key)) {
			return htReg.get(index).get(key);
		} else {
			return Pair.createPair((float)0.0, (float)0.0);
		}
	}
	
	String getCls(int key) {
		if (htCls.containsKey(key)) {
			return htCls.get(key);
		} else {
			return "";
		}
	}
	
	int getBlockTrainNumSum(int block) {
		if (block < trainRegTotalNum.length) {
			return trainRegTotalNum[block];
		} else { 
			return 0;
		}
	}
}
