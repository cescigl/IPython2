import org.apache.hadoop.hive.ql.exec.UDF;

import eu.bitwalker.useragentutils.UserAgent;

public class ParseUserAgent_UDF extends UDF {
	public String evaluate(final String userAgent) {
		StringBuilder builder = new StringBuilder();
		UserAgent ua = new UserAgent(userAgent);
		builder.append(ua.getOperatingSystem() + "\t" + ua.getBrowser() + "\t" + ua.getBrowserVersion());
		return builder.toString();
	}
	
	public static void main(String args[]) {
		ParseUserAgent_UDF pudf = new ParseUserAgent_UDF();
		String str=pudf.evaluate("Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.28) Gecko/20120306 Firefox/3.6.28");
		System.out.println(str);
		System.out.println(pudf.evaluate("Mozilla/5.0 (Windows NT 6.1; rv:37.0) Gecko/20100101 Firefox/37.0"));
		System.out.println(pudf.evaluate("Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36"));
		System.out.println(pudf.evaluate("Mozilla/5.0 (Windows NT 6.1; rv:40.0) Gecko/20100101 Firefox/40.0"));
		System.out.println(pudf.evaluate("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36"));
		System.out.println(pudf.evaluate(""));
		System.out.println(pudf.evaluate("abcdcwd 4 5 6 "));
		//System.out.println(pudf.evaluate(""));
		//System.out.println(pudf.evaluate(""));
	}

}