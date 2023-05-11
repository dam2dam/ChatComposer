package com.a504.chatcomposer.produce.dto.response;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;

@Getter
@Setter
@NoArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class FileResp extends BaseResponseBody {
    private byte[] mixedMusicWav;

    public static FileResp of(String msg, Integer statusCode, byte[] mixedMusicWav) {
        FileResp res = new FileResp();
        res.setMsg(msg);
        res.setStatusCode(statusCode);
        res.setMixedMusicWav(mixedMusicWav);
        return res;
    }

}
