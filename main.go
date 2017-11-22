package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"
)

func main() {

	fmt.Println("Iniciando...")
	type Video struct {
		Titulo  string
		ID      string
		Duracao string
		URL     string
		Fase    int32
	}

	var jason map[string]Video

	var re = regexp.MustCompile(`(?s)div[^\>]+data-context-item-id\=\"(?P<idzinho>\w+)\"[^\>]+\>\s+<div[^\>]+\>\s+<div[^\>]+\>.*?<a\shref="(?P<href1>.*?)"[^\>]+\>.*?class="video-time".*?\>.*?\>(?P<tempo>[\d\:]+)\<\/span>.*?div class.*?<h3 class.*?a class.*?title="(?P<title1>.*?)".*?href="(?P<href2>.*?)".*?>(?P<title2>.*?)<\/a>`)
	resp, err := http.Get("https://www.youtube.com/channel/UCWijW6tW0iI5ghsAbWDFtTg/videos")
	if err != nil {
		fmt.Println("Erro")
		fmt.Println(err)
		return
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	jason = make(map[string]Video)
	for i, match := range re.FindAllStringSubmatch(string(body), -1) {
		if strings.Contains(match[4], "É da Coisa") {
			fmt.Println(match[1:], "found at index", i, " --> ", len(match[3]))
			info := Video{
				match[4],
				match[1],
				match[3],
				match[2],
				0,
			}
			jason[match[1]] = info
		}
	}

	fmt.Println()
	fmt.Println(jason)

	jsonData, _ := json.Marshal(jason)

	ioutil.WriteFile("edacoisa.json", jsonData, 0666)
	fmt.Println(string(jsonData))

}
