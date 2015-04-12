package magPadReference;

public class Pair<K, V> {
	private final K x;    
	private final V y;
	    
	public static <K, V> Pair<K, V> createPair(K element0, V element1) {        
		return new Pair<K, V>(element0, element1);
	}
	    
	public Pair(K element0, V element1) {
		this.x = element0;
		this.y = element1;   
	}
	
	public K getX() {
		return x;   
	}
	
	public V getY() {
		return y;   
	}
}
