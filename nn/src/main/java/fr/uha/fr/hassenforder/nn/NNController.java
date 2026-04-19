package fr.uha.fr.hassenforder.nn;

import java.util.Map;
import java.util.TreeMap;
import javax.ws.rs.QueryParam;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class NNController {

    @PostMapping (value="/config")
    @ResponseStatus(HttpStatus.OK)
    public String config (@QueryParam( "algo" ) String algo) {
        
        String url = "http://ia:80/config?algo={algo}";

        HttpHeaders headers = new HttpHeaders();
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(headers);
        
        Map<String, String> params = new TreeMap<>();
        params.put("algo", algo);

        RestTemplate template = new RestTemplate();
        ResponseEntity<String> response = template.exchange(url, HttpMethod.POST, requestEntity, String.class, params);

        return response.getBody();
    }

    @PostMapping (value="/classify")
    @ResponseStatus(HttpStatus.OK)
    public String classify (@RequestParam( "datafile" ) MultipartFile datafile) {

        String url = "http://ia:80/classify";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("datafile", datafile.getResource());
        
        Map<String, String> params = new TreeMap<>();

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        
        RestTemplate template = new RestTemplate();
        ResponseEntity<String> response = template.exchange(url, HttpMethod.POST, requestEntity, String.class, params);

        return response.getBody();
    }
}
